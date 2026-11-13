"""Versioned dependency manifests and pre-checked loading for cloudpickle.

This module provides :func:`dumps_with_manifest` and :func:`loads_checked`,
an opt-in layer on top of the regular :func:`cloudpickle.dumps` /
:func:`cloudpickle.loads` pair.  Alongside the pickle payload, a structured
manifest records the decisions actually taken by the reducers while
serializing:

* the by-reference ``(module, qualname)`` lookups that the loader will have
  to resolve through the import machinery;
* the submodules that cloudpickle explicitly discovered for by-value
  functions (``_find_imported_submodules``);
* the Python implementation / version and the bytecode ABI identity
  (``importlib.util.MAGIC_NUMBER``) required to rebuild code objects carried
  by value;
* the opaque boundaries introduced by custom ``persistent_id`` hooks or by
  user supplied reconstructors (``__reduce__`` / ``__reduce_ex__``), whose
  internal dynamic imports cannot be discovered from the outside;
* optionally, the installed distribution version and a digest of the
  explicit module file of each referenced module.

The manifest is produced by symbolically walking the emitted pickle opcode
stream with :mod:`pickletools` and by observing the pickler's own reduction
calls.  It never scans source strings to guess imports, never invokes a
user ``__reduce__`` a second time and never imports a candidate module on
the serializing side as a probe: only modules already present in
``sys.modules`` are introspected.

Security and compatibility caveats
-----------------------------------

A matching manifest does **not** make an untrusted pickle safe: unpickling
still executes reconstructor callables.  A manifest only lets the caller
check, *before* any pickle reconstruction starts, that the payload has not
been tampered with and that the local environment matches the compatibility
policy the caller has explicitly chosen.

This module does not promise long-term compatibility across Python
versions.  In particular, code objects pickled by value are tied to the
bytecode ABI of the serializing interpreter.

No dependency is ever installed or downloaded automatically.  Environment
mismatches are returned as structured differences that the caller may
accept explicitly through ``allowed_differences``.

The manifest never contains module globals' values, file paths or
credential paths: only module / qualname strings, distribution metadata and
non-reversible file digests are recorded.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as _metadata
import importlib.util
import io
import json
import pickle
import pickletools
import platform
import struct
import sys

from .cloudpickle import Pickler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_VERSION = 1

# Fixed 8-byte magic prefixing every framed envelope.
_MANIFEST_MAGIC = b"CPCKMNF1"
_HEADER_STRUCT = struct.Struct(">Q")

_DIGEST_ALGORITHM = "sha256"

# Difference kinds that :func:`loads_checked` know how to report.
DIFF_MANIFEST_FORMAT = "manifest_format"
DIFF_PAYLOAD_DIGEST = "payload_digest"
DIFF_PYTHON_IMPLEMENTATION = "python_implementation"
DIFF_PYTHON_VERSION = "python_version"
DIFF_CACHE_TAG = "cache_tag"
DIFF_BYTECODE_ABI = "bytecode_abi"
DIFF_MODULE_MISSING = "module_missing"
DIFF_QUALNAME_MISSING = "qualname_missing"
DIFF_MODULE_KIND = "module_kind"
DIFF_MODULE_FILE_MISSING = "module_file_missing"
DIFF_DISTRIBUTION_VERSION = "distribution_version"
DIFF_FILE_DIGEST = "file_digest"
DIFF_OPAQUE_BOUNDARY = "opaque_boundary"

MANIFEST_DIFFERENCE_KINDS = frozenset(
    {
        DIFF_MANIFEST_FORMAT,
        DIFF_PAYLOAD_DIGEST,
        DIFF_PYTHON_IMPLEMENTATION,
        DIFF_PYTHON_VERSION,
        DIFF_CACHE_TAG,
        DIFF_BYTECODE_ABI,
        DIFF_MODULE_MISSING,
        DIFF_QUALNAME_MISSING,
        DIFF_MODULE_KIND,
        DIFF_MODULE_FILE_MISSING,
        DIFF_DISTRIBUTION_VERSION,
        DIFF_FILE_DIGEST,
        DIFF_OPAQUE_BOUNDARY,
    }
)

# Module names under which cloudpickle's own reconstructors can be referenced
# in pickle streams (the ``cloudpickle_fast`` alias is kept for backward
# compatibility by the package).
_CLOUDPICKLE_MODULE_NAMES = frozenset(
    {"cloudpickle", "cloudpickle.cloudpickle", "cloudpickle.cloudpickle_fast"}
)
_CLOUDPICKLE_ROOT = "cloudpickle"

# Fallback for Python < 3.10 where ``sys.stdlib_module_names`` does not exist.
_FALLBACK_STDLIB_ROOTS = frozenset(
    {
        "_collections_abc",
        "abc",
        "builtins",
        "collections",
        "copyreg",
        "dataclasses",
        "enum",
        "functools",
        "io",
        "itertools",
        "logging",
        "os",
        "pickle",
        "sys",
        "types",
        "typing",
        "weakref",
    }
)
_STDLIB_ROOTS = frozenset(getattr(sys, "stdlib_module_names", _FALLBACK_STDLIB_ROOTS))

# Opcodes pushing an immediate scalar value (the decoded scalar is retained
# in the symbolic stack so that module / name strings can be resolved).
_SCALAR_PUSH_OPCODES = frozenset(
    {
        "INT",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "FLOAT",
        "BINFLOAT",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BINBYTES8",
        "BYTEARRAY8",
        "UNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
    }
)

_MARK_BASED_OPCODES = frozenset({"LIST", "DICT", "SET", "FROZENSET"})


# ---------------------------------------------------------------------------
# Errors and reports
# ---------------------------------------------------------------------------


class ManifestError(Exception):
    """Base class for manifest related errors."""


class ManifestFormatError(ManifestError):
    """Raised when an envelope or its manifest cannot be parsed/validated.

    This error is always raised before any pickle reconstruction starts.
    """


class ManifestCheckError(ManifestError):
    """Raised when a checked environment does not satisfy the policy."""

    def __init__(self, report):
        self.report = report
        blocking = report.blocking
        lines = [
            "manifest check failed with {n} blocking difference(s):".format(
                n=len(blocking)
            )
        ]
        for difference in blocking:
            lines.append(
                "  - [{kind}] {subject}: {message}".format(
                    kind=difference["kind"],
                    subject=difference.get("subject", ""),
                    message=difference.get("message", ""),
                )
            )
        super().__init__("\n".join(lines))


class ManifestCheckReport:
    """Result of :func:`verify_manifest`.

    Attributes
    ----------
    ok:
        True when no difference outside of ``allowed_differences`` was found.
    differences:
        All structured differences observed during the precheck, including
        the ones explicitly allowed by the caller.
    blocking:
        Differences that are not allowed and therefore prevent reconstruction.
    allowed:
        Frozen set of difference kinds the caller explicitly accepts.
    envelope:
        The parsed :class:`ManifestEnvelope` that was checked.
    """

    def __init__(self, envelope, differences, allowed):
        self.envelope = envelope
        self.differences = differences
        self.allowed = frozenset(allowed)
        self.blocking = [d for d in differences if d["kind"] not in self.allowed]
        self.ok = not self.blocking

    def __repr__(self):
        return (
            "ManifestCheckReport(ok={ok}, n_differences={n}, "
            "n_blocking={m})".format(
                ok=self.ok, n=len(self.differences), m=len(self.blocking)
            )
        )


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class ManifestEnvelope:
    """A manifest together with the pickle payload it describes.

    The binary framing is intentionally simple and does not rely on pickle
    itself, so an envelope can be validated without ever executing any
    reconstructor::

        magic (8 bytes) | manifest length (uint64 BE) | JSON manifest | payload
    """

    __slots__ = ("manifest", "payload")

    def __init__(self, manifest, payload):
        self.manifest = manifest
        self.payload = bytes(payload)

    # -- (de)serialization --------------------------------------------------

    def to_bytes(self):
        """Return the framed binary representation of the envelope."""
        body = json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return _MANIFEST_MAGIC + _HEADER_STRUCT.pack(len(body)) + body + self.payload

    @classmethod
    def from_bytes(cls, data):
        """Parse a framed envelope without touching the pickle payload."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ManifestFormatError(
                "envelope must be bytes-like, got {}".format(type(data).__name__)
            )
        data = bytes(data)
        header_size = len(_MANIFEST_MAGIC) + _HEADER_STRUCT.size
        if len(data) < header_size:
            raise ManifestFormatError("envelope is truncated")
        if data[: len(_MANIFEST_MAGIC)] != _MANIFEST_MAGIC:
            raise ManifestFormatError(
                "envelope does not start with the cloudpickle manifest magic"
            )
        (length,) = _HEADER_STRUCT.unpack(data[len(_MANIFEST_MAGIC) : header_size])
        if len(data) < header_size + length:
            raise ManifestFormatError(
                "envelope is truncated: declared manifest is larger than the "
                "envelope itself"
            )
        raw_manifest = data[header_size : header_size + length]
        payload = data[header_size + length :]
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise ManifestFormatError("manifest is not valid UTF-8") from e
        except json.JSONDecodeError as e:
            raise ManifestFormatError(
                "manifest is not valid JSON: {}".format(e.msg)
            ) from e
        cls.validate_manifest_schema(manifest)
        return cls(manifest, payload)

    @staticmethod
    def validate_manifest_schema(manifest):
        """Validate the structural shape of a manifest.

        Content validation against the local environment happens in
        :func:`verify_manifest`; this method only enforces the format
        contract so that a malformed manifest can never reach the pickle
        reconstruction stage.
        """
        if not isinstance(manifest, dict):
            raise ManifestFormatError("manifest root must be a JSON object")
        if manifest.get("manifest_version") != MANIFEST_VERSION:
            raise ManifestFormatError(
                "unsupported manifest_version: {!r}, expected {}".format(
                    manifest.get("manifest_version"), MANIFEST_VERSION
                )
            )
        required = ("payload", "python", "references", "modules", "dynamic")
        for key in required:
            if key not in manifest:
                raise ManifestFormatError(
                    "manifest is missing required field {!r}".format(key)
                )

        payload_section = manifest["payload"]
        if not isinstance(payload_section, dict) or not isinstance(
            payload_section.get("digest"), str
        ):
            raise ManifestFormatError("manifest.payload.digest must be a string")
        if not isinstance(payload_section.get("algorithm"), str):
            raise ManifestFormatError("manifest.payload.algorithm must be a string")

        python_section = manifest["python"]
        if not isinstance(python_section, dict):
            raise ManifestFormatError("manifest.python must be an object")
        for key in (
            "implementation",
            "version_info",
            "cache_tag",
            "magic_number",
        ):
            if key not in python_section:
                raise ManifestFormatError("manifest.python is missing {!r}".format(key))
        if not isinstance(python_section["version_info"], list):
            raise ManifestFormatError("manifest.python.version_info must be a list")

        if not isinstance(manifest["references"], list):
            raise ManifestFormatError("manifest.references must be a list")
        for reference in manifest["references"]:
            if (
                not isinstance(reference, dict)
                or not isinstance(reference.get("module"), str)
                or not isinstance(reference.get("qualname"), str)
            ):
                raise ManifestFormatError(
                    "each reference must be an object with string "
                    "'module' and 'qualname'"
                )

        if not isinstance(manifest["modules"], dict):
            raise ManifestFormatError("manifest.modules must be an object")
        for name, entry in manifest["modules"].items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise ManifestFormatError(
                    "each module entry must map a name to an object"
                )
            if not isinstance(entry.get("kind"), str):
                raise ManifestFormatError(
                    "module entry {!r} is missing a string 'kind'".format(name)
                )

        dynamic = manifest["dynamic"]
        if not isinstance(dynamic, dict) or not isinstance(
            dynamic.get("code_object_count"), int
        ):
            raise ManifestFormatError(
                "manifest.dynamic.code_object_count must be an integer"
            )

        opaque = manifest.get("opaque", [])
        if not isinstance(opaque, list):
            raise ManifestFormatError("manifest.opaque must be a list")
        for entry in opaque:
            if not isinstance(entry, dict) or not isinstance(entry.get("kind"), str):
                raise ManifestFormatError(
                    "each opaque entry must be an object with a string 'kind'"
                )

    def to_dict(self):
        """Return a JSON-serializable view of the envelope."""
        return {"manifest": self.manifest, "payload": self.payload.hex()}


# ---------------------------------------------------------------------------
# Recording pickler
# ---------------------------------------------------------------------------


class ManifestPickler(Pickler):
    """Cloudpickle pickler that additionally observes reduction decisions.

    This pickler behaves exactly like :class:`cloudpickle.Pickler`: no
    reducer is invoked twice and no object is introspected outside of the
    normal pickling pass.  It only records two things that are not directly
    visible from the emitted stream:

    * objects handled through a ``persistent_id`` hook (opaque persistent
      boundary);
    * the submodules explicitly discovered by cloudpickle for functions
      pickled by value.
    """

    def __init__(self, file, protocol=None, buffer_callback=None, persistent_id=None):
        super().__init__(file, protocol=protocol, buffer_callback=buffer_callback)
        self._manifest_persistent_id = persistent_id
        self.manifest_persistent = []
        self.manifest_submodules = set()
        self.manifest_dynamic_functions = []

    def persistent_id(self, obj):
        if self._manifest_persistent_id is None:
            return None
        persistent_id = self._manifest_persistent_id(obj)
        if persistent_id is not None:
            # Only record the object type name: never the id itself, which
            # may carry arbitrary user data.
            self.manifest_persistent.append(type(obj).__name__)
        return persistent_id

    def _dynamic_function_reduce(self, func):
        reduced = super()._dynamic_function_reduce(func)
        # ``reduced`` is the single reduction tuple also used by the pickler
        # to emit the payload: observing it costs no extra reducer call.
        # Its state is ``(state_dict, slotstate)`` and the slotstate carries
        # the submodules that cloudpickle explicitly discovered.
        try:
            slotstate = reduced[2][1]
            for submodule in slotstate.get("_cloudpickle_submodules", ()):
                name = getattr(submodule, "__name__", None)
                if name is not None:
                    self.manifest_submodules.add(name)
            self.manifest_dynamic_functions.append(
                "{module}.{qualname}".format(
                    module=getattr(func, "__module__", "?"),
                    qualname=getattr(func, "__qualname__", "?"),
                )
            )
        except (IndexError, KeyError, TypeError):  # pragma: no cover
            # Defensive: never let manifest bookkeeping break pickling.
            pass
        return reduced


# ---------------------------------------------------------------------------
# Symbolic pickle stream walk
# ---------------------------------------------------------------------------


def _parse_global_arg(arg):
    """Parse a GLOBAL/INST argument into a ``(module, qualname)`` pair.

    Modern protocols emit the pair as a tuple while protocols 0 and 1 emit
    a single ``"module qualname"`` string.
    """
    if isinstance(arg, tuple):
        return arg[0], arg[1]
    module, qualname = arg.split(" ", 1)
    return module, qualname


def _tuple_marker(items):
    """Collapse a sequence of markers into a retained-value tuple marker.

    Scalar slots retain their value (typically strings), which is enough to
    resolve the first argument of ``subimport`` / ``dynamic_subimport``
    reductions even when later slots are non-scalar objects such as a dynamic
    module's state dict.  Non-scalar slots are represented by ``None``.
    """
    return (
        "t",
        tuple(item[1] if item[0] == "s" else None for item in items),
    )


def _walk_pickle_stream(payload):
    """Walk pickle opcodes symbolically, without reconstructing anything.

    Returns a dict describing the reduction decisions that were actually
    emitted into the stream.

    Symbolic stack markers are tuples:

    * ``("s", value)``: an immediate scalar (string / number / ...);
    * ``("t", values)``: a tuple whose scalar contents are retained;
    * ``("g", module, qualname)``: a global reference;
    * ``("prov", category)``: the result of a reduction, where category is
      ``"internal"`` (stdlib or cloudpickle governed) or ``"opaque"`` (a user
      reducer boundary);
    * ``("v",)``: any other value;
    * ``("mark",)``: a pickle mark.
    """
    references = set()
    subimports = set()
    dynamic_modules = set()
    opaque = []
    opaque_seen = set()
    code_object_count = 0
    dynamic_class_count = 0
    dynamic_function_count = 0

    stack = []
    memo = {}
    memo_counter = 0
    mark = ("mark",)
    internal = ("prov", "internal")
    opaque_prov = ("prov", "opaque")

    def record_opaque(kind, **details):
        key = (kind, tuple(sorted(details.items())))
        if key not in opaque_seen:
            opaque_seen.add(key)
            entry = {"kind": kind}
            entry.update(details)
            opaque.append(entry)

    def classify_callable(callable_marker, args_marker):
        """Classify a reduce callable and return its result provenance.

        Also updates the dedicated counters for cloudpickle decisions
        (submodules, dynamic modules / classes / functions, code objects).
        """
        nonlocal code_object_count, dynamic_class_count, dynamic_function_count

        if callable_marker[0] == "prov" or callable_marker[0] != "g":
            # Any non-global callable (including a function carried by value
            # and rebuilt through cloudpickle's own reconstructors) is
            # user-supplied behavior used as a reducer: nothing can be known
            # about its internal imports at load time. The only internal
            # reduction whose result is later called is CodeType, which is
            # aliased to the types.CodeType global above.
            record_opaque("dynamic_reconstructor")
            return opaque_prov

        module, qualname = callable_marker[1], callable_marker[2]

        if (
            module in _CLOUDPICKLE_MODULE_NAMES
            and qualname == "subimport"
            and args_marker
            and args_marker[0] == "t"
        ):
            subimports.add(args_marker[1][0])
            return internal
        if (
            module in _CLOUDPICKLE_MODULE_NAMES
            and qualname == "dynamic_subimport"
            and args_marker
            and args_marker[0] == "t"
        ):
            dynamic_modules.add(args_marker[1][0])
            return internal
        if module == "types" and qualname == "CodeType":
            # Direct construction of a code object (older cloudpickle streams
            # reduce CodeType itself).
            code_object_count += 1
            return internal
        if (
            module in _CLOUDPICKLE_MODULE_NAMES
            and qualname == "_builtin_type"
            and args_marker
            and args_marker[0] == "t"
            and args_marker[1]
            and args_marker[1][0] == "CodeType"
        ):
            # Current cloudpickle streams first fetch the CodeType type via
            # _builtin_type("CodeType"); alias the result to the types.CodeType
            # global so that the following CodeType(...) construction is
            # counted exactly once.
            return ("g", "types", "CodeType")
        if module in _CLOUDPICKLE_MODULE_NAMES and qualname in (
            "_make_skeleton_class",
            "_make_skeleton_enum",
        ):
            dynamic_class_count += 1
            return internal
        if module in _CLOUDPICKLE_MODULE_NAMES and qualname == "_make_function":
            dynamic_function_count += 1
            return internal

        root = module.split(".", 1)[0]
        if root in _STDLIB_ROOTS or root == _CLOUDPICKLE_ROOT:
            # Reconstruction governed by the standard library or by
            # cloudpickle itself: no opaque user boundary.
            return internal

        record_opaque(
            "external_reconstructor",
            reconstructor="{module}.{qualname}".format(
                module=module, qualname=qualname
            ),
        )
        return opaque_prov

    def pop_to_mark():
        items = []
        while True:
            marker = stack.pop()
            if marker is mark:
                return items
            items.append(marker)

    for opcode, arg, _position in pickletools.genops(io.BytesIO(payload)):
        name = opcode.name

        if name in ("PROTO", "FRAME"):
            continue
        if name == "STOP":
            break
        if name == "MARK":
            stack.append(mark)
        elif name in _SCALAR_PUSH_OPCODES:
            stack.append(("s", arg))
        elif name == "NONE":
            stack.append(("s", None))
        elif name == "NEWTRUE":
            stack.append(("s", True))
        elif name == "NEWFALSE":
            stack.append(("s", False))
        elif name == "EMPTY_TUPLE":
            stack.append(("t", ()))
        elif name in ("EMPTY_LIST", "EMPTY_DICT", "EMPTY_SET"):
            stack.append(("v",))
        elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
            size = int(name[-1])
            items = [stack.pop() for _ in range(size)][::-1]
            stack.append(_tuple_marker(items))
        elif name == "TUPLE":
            stack.append(_tuple_marker(pop_to_mark()))
        elif name in _MARK_BASED_OPCODES:
            pop_to_mark()
            stack.append(("v",))
        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            memo[arg] = stack[-1]
        elif name == "MEMOIZE":
            memo[memo_counter] = stack[-1]
            memo_counter += 1
        elif name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo[arg])
        elif name == "GLOBAL":
            module, qualname = _parse_global_arg(arg)
            references.add((module, qualname))
            stack.append(("g", module, qualname))
        elif name == "STACK_GLOBAL":
            qualname_marker = stack.pop()
            module_marker = stack.pop()
            module = module_marker[1]
            qualname = qualname_marker[1]
            references.add((module, qualname))
            stack.append(("g", module, qualname))
        elif name == "REDUCE":
            args_marker = stack.pop()
            callable_marker = stack.pop()
            stack.append(classify_callable(callable_marker, args_marker))
        elif name == "NEWOBJ":
            stack.pop()  # args
            class_marker = stack.pop()
            stack.append(classify_callable(class_marker, None))
        elif name == "NEWOBJ_EX":
            stack.pop()  # kwargs
            stack.pop()  # args
            class_marker = stack.pop()
            stack.append(classify_callable(class_marker, None))
        elif name == "OBJ":
            pop_to_mark()
            class_marker = stack.pop()
            stack.append(classify_callable(class_marker, None))
        elif name == "INST":
            pop_to_mark()
            module, qualname = _parse_global_arg(arg)
            references.add((module, qualname))
            stack.append(classify_callable(("g", module, qualname), None))
        elif name == "POP":
            stack.pop()
        elif name == "POP_MARK":
            pop_to_mark()
        elif name == "BUILD":
            # Per the pickletools stack model both the object and its state
            # are consumed; further references go through the memo.
            stack.pop()
            stack.pop()
        elif name in ("SETITEM", "DICT_SETITEM"):
            # Key and value are consumed; the dict marker stays on the stack.
            stack.pop()
            stack.pop()
        elif name in ("APPEND", "LIST_APPEND"):
            # The appended item is consumed; the list marker stays.
            stack.pop()
        elif name in ("SETITEMS", "APPENDS", "ADDITEMS"):
            # Everything up to the mark is consumed; the container marker
            # stays on the stack.
            pop_to_mark()
        elif name in ("NEXT_BUFFER", "READONLY_BUFFER"):
            stack.append(("v",))
        else:
            # Forward-compatible fallback: honor the declared stack effect
            # of any opcode introduced by a newer Python version.
            before = opcode.stack_before
            after = opcode.stack_after
            if "mark" in before:
                pop_to_mark()
                for _ in before:
                    if _ != "mark" and _ != "stackslice":
                        stack.pop()
            else:
                for _ in before:
                    stack.pop()
            for _ in after:
                stack.append(("v",))

    return {
        "references": references,
        "subimports": subimports,
        "dynamic_modules": dynamic_modules,
        "opaque": opaque,
        "code_object_count": code_object_count,
        "dynamic_class_count": dynamic_class_count,
        "dynamic_function_count": dynamic_function_count,
    }


# ---------------------------------------------------------------------------
# Module introspection (serializing side: sys.modules only, never import)
# ---------------------------------------------------------------------------


def _module_kind(module):
    """Classify an already-imported module without importing anything."""
    spec = getattr(module, "__spec__", None)
    if spec is None:
        # No import machinery record: the module was created at runtime
        # (types.ModuleType) or injected by an extension module.
        if hasattr(module, "__path__"):
            return "namespace"
        return "dynamic"

    origin = getattr(spec, "origin", None)
    if origin == "built-in":
        return "builtin"
    if origin == "frozen":
        return "frozen"
    if origin is None:
        if getattr(spec, "submodule_search_locations", None) is not None:
            return "namespace"
        return "dynamic"
    if getattr(spec, "submodule_search_locations", None) is not None:
        return "package"
    return "module"


class _LazyDistributionMapping:
    """Lazily computed, single-use ``packages_distributions`` cache.

    A full scan of every installed distribution is expensive, so it is only
    triggered when a cheap direct lookup fails, and only once per manifest
    dump or check.
    """

    def __init__(self):
        self._mapping = {}
        self._resolved = False

    def get(self):
        if not self._resolved:
            self._resolved = True
            if hasattr(_metadata, "packages_distributions"):
                try:
                    self._mapping = _metadata.packages_distributions()
                except Exception:
                    self._mapping = {}
        return self._mapping


def _distribution_for(module_name, mapping):
    """Best-effort installed distribution metadata for a module name.

    Only reads installed distribution metadata (``.dist-info``); never
    imports the candidate module and never accesses the network.  ``mapping``
    is a per-call :class:`_LazyDistributionMapping`.
    """
    root = module_name.split(".", 1)[0]
    try:
        # Fast path: the distribution name usually matches the import name.
        version = _metadata.version(root)
        return {"name": root, "version": version}
    except _metadata.PackageNotFoundError:
        pass
    except Exception:
        return None

    # Slow path for packages whose distribution name differs from their
    # import name; the full mapping is scanned at most once per call.
    names = mapping.get().get(root)
    if not names:
        return None
    distribution_name = names[0]
    try:
        version = _metadata.version(distribution_name)
    except Exception:
        return None
    return {"name": distribution_name, "version": version}


def _digest_file(path):
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None
    return {"algorithm": _DIGEST_ALGORITHM, "digest": digest}


def _describe_module(
    name, include_distributions, include_file_digests, distribution_mapping
):
    entry = {"kind": "unobserved", "distribution": None, "file_digest": None}
    module = sys.modules.get(name)
    if module is None:
        # The name was emitted by a reducer but the module is not currently
        # importable in the serializing process: do not import it just to
        # inspect it.
        return entry

    kind = _module_kind(module)
    entry["kind"] = kind

    if include_distributions and kind not in ("builtin", "frozen", "dynamic"):
        # Built-in, frozen and runtime-created modules are never installed
        # distributions: skip metadata lookups (and their possible fallback
        # scans) for them.
        entry["distribution"] = _distribution_for(name, distribution_mapping)

    if include_file_digests:
        file_path = getattr(module, "__file__", None)
        if file_path is not None:
            # Only the digest is recorded: the path itself may contain user
            # or environment specific information and is never stored.
            entry["file_digest"] = _digest_file(file_path)

    return entry


def _python_identity(bytecode_dynamic, code_object_count):
    magic_number = int.from_bytes(importlib.util.MAGIC_NUMBER, "little")
    return {
        "implementation": platform.python_implementation(),
        "version_info": list(sys.version_info),
        "cache_tag": sys.implementation.cache_tag,
        "magic_number": magic_number,
        "abi_flags": getattr(sys, "abiflags", ""),
        "bytecode_dynamic": bytecode_dynamic,
        "code_object_count": code_object_count,
    }


# ---------------------------------------------------------------------------
# Public dumping API
# ---------------------------------------------------------------------------


def dumps_with_manifest(
    obj,
    protocol=None,
    *,
    include_distributions=True,
    include_file_digests=False,
    persistent_id=None,
    buffer_callback=None,
):
    """Serialize ``obj`` and build a versioned dependency manifest.

    The pickle payload is produced by a single, standard cloudpickle
    reduction pass: recording the manifest never triggers a second call to
    any user defined ``__reduce__`` / ``__reduce_ex__`` and never imports a
    module to probe it.

    Parameters
    ----------
    obj:
        Object to serialize.
    protocol:
        Pickle protocol, forwarded to :class:`cloudpickle.Pickler`.
    include_distributions:
        When True (the default), record the installed distribution version
        of referenced modules when distribution metadata is available.
    include_file_digests:
        When True, record a SHA-256 digest of the explicit ``__file__`` of
        each file-backed referenced module.  Paths themselves are never
        recorded.  Disabled by default as hashing large package trees can be
        expensive.
    persistent_id:
        Optional persistent id callable, as supported by ``pickle``.  Objects
        handled through it are flagged as opaque boundaries in the manifest.
    buffer_callback:
        Forwarded to :class:`cloudpickle.Pickler` for out-of-band buffers.

    Returns
    -------
    ManifestEnvelope
    """
    with io.BytesIO() as file:
        pickler = ManifestPickler(
            file,
            protocol=protocol,
            buffer_callback=buffer_callback,
            persistent_id=persistent_id,
        )
        pickler.dump(obj)
        payload = file.getvalue()

    analysis = _walk_pickle_stream(payload)

    module_names = set()
    for module_name, _qualname in analysis["references"]:
        module_names.add(module_name)
    module_names |= analysis["subimports"]
    module_names |= analysis["dynamic_modules"]
    module_names |= pickler.manifest_submodules

    distribution_mapping = _LazyDistributionMapping() if include_distributions else None
    modules = {}
    for name in sorted(module_names):
        modules[name] = _describe_module(
            name,
            include_distributions=include_distributions,
            include_file_digests=include_file_digests,
            distribution_mapping=distribution_mapping,
        )

    # A module reduced through ``dynamic_subimport`` is, by the reducer's own
    # decision, carried by value as a fileless dynamic module: honor that
    # decision even when it is not present in ``sys.modules`` at dump time.
    for name in analysis["dynamic_modules"]:
        if modules[name]["kind"] == "unobserved":
            modules[name]["kind"] = "dynamic"

    references = [
        {"module": module_name, "qualname": qualname}
        for module_name, qualname in sorted(analysis["references"])
    ]

    opaque = list(analysis["opaque"])
    for object_type in pickler.manifest_persistent:
        opaque.append({"kind": "persistent_id", "object_type": object_type})

    code_object_count = analysis["code_object_count"]
    from . import __version__ as cloudpickle_version

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "cloudpickle_version": cloudpickle_version,
        "payload": {
            "algorithm": _DIGEST_ALGORITHM,
            "digest": hashlib.sha256(payload).hexdigest(),
        },
        "python": _python_identity(code_object_count > 0, code_object_count),
        "references": references,
        "modules": modules,
        "submodules": sorted(pickler.manifest_submodules),
        "dynamic": {
            "functions": sorted(pickler.manifest_dynamic_functions),
            "dynamic_function_count": analysis["dynamic_function_count"],
            "dynamic_class_count": analysis["dynamic_class_count"],
            "code_object_count": code_object_count,
            "modules": sorted(analysis["dynamic_modules"]),
            "bytecode_dynamic": code_object_count > 0,
        },
        "opaque": opaque,
    }
    return ManifestEnvelope(manifest=manifest, payload=payload)


# ---------------------------------------------------------------------------
# Checking side
# ---------------------------------------------------------------------------


def _difference(kind, subject, expected, actual, message):
    return {
        "kind": kind,
        "subject": subject,
        "expected": expected,
        "actual": actual,
        "message": message,
    }


def _normalize_policy(policy, allowed_differences):
    if policy != "strict":
        raise ValueError(
            "unknown policy {!r}; only 'strict' is supported, combined with "
            "an explicit allowed_differences set".format(policy)
        )
    if allowed_differences is None:
        return frozenset()
    allowed = frozenset(allowed_differences)
    unknown = allowed - MANIFEST_DIFFERENCE_KINDS
    if unknown:
        raise ValueError(
            "unknown difference kind(s) in allowed_differences: {}".format(
                sorted(unknown)
            )
        )
    return allowed


def _locate_module_without_executing(name):
    """Locate ``name`` through the import machinery without executing it.

    Returns ``(module_or_None, spec_or_None)``.  A module already present in
    ``sys.modules`` is returned directly.  Otherwise :func:`find_spec` is
    used: it locates the module without executing its code (note that, as
    with any import machinery operation, parent packages of a submodule may
    be imported).  Returns ``(None, None)`` when the module cannot be found.
    """
    module = sys.modules.get(name)
    if module is not None:
        return module, getattr(module, "__spec__", None)
    try:
        spec = importlib.util.find_spec(name)
    except (ModuleNotFoundError, ValueError):
        return None, None
    if spec is None:
        return None, None
    return None, spec


def _resolve_qualname(module, qualname):
    obj = module
    for part in qualname.split("."):
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    return True


def _check_reference(differences, module_name, qualname):
    module, spec = _locate_module_without_executing(module_name)
    if module is None and spec is None:
        differences.append(
            _difference(
                DIFF_MODULE_MISSING,
                module_name,
                "importable module",
                "not found",
                "module {!r} referenced by the manifest is not importable".format(
                    module_name
                ),
            )
        )
        return
    if module is not None and not _resolve_qualname(module, qualname):
        differences.append(
            _difference(
                DIFF_QUALNAME_MISSING,
                "{}.{}".format(module_name, qualname),
                "attribute resolvable",
                "attribute missing",
                "qualname {!r} cannot be resolved in module {!r}".format(
                    qualname, module_name
                ),
            )
        )


def _check_module_entry(differences, name, entry, distribution_mapping):
    recorded_kind = entry.get("kind")
    module = sys.modules.get(name)

    if recorded_kind not in ("unobserved", "dynamic"):
        if module is not None:
            actual_kind = _module_kind(module)
            if actual_kind != recorded_kind:
                differences.append(
                    _difference(
                        DIFF_MODULE_KIND,
                        name,
                        recorded_kind,
                        actual_kind,
                        "module kind changed since serialization",
                    )
                )
        else:
            _located, spec = _locate_module_without_executing(name)
            if spec is None:
                differences.append(
                    _difference(
                        DIFF_MODULE_MISSING,
                        name,
                        "importable module",
                        "not found",
                        "module {!r} is no longer importable".format(name),
                    )
                )

    distribution = entry.get("distribution")
    if distribution:
        actual_distribution = _distribution_for(name, distribution_mapping)
        actual_version = actual_distribution["version"] if actual_distribution else None
        if actual_version != distribution["version"]:
            differences.append(
                _difference(
                    DIFF_DISTRIBUTION_VERSION,
                    name,
                    distribution["version"],
                    actual_version,
                    "distribution {!r} version differs".format(distribution["name"]),
                )
            )

    recorded_digest = entry.get("file_digest")
    if recorded_digest:
        file_path = None
        if module is not None:
            file_path = getattr(module, "__file__", None)
        if file_path is None:
            _located, spec = _locate_module_without_executing(name)
            if spec is not None:
                origin = getattr(spec, "origin", None)
                if origin and origin != "built-in" and origin != "frozen":
                    file_path = origin
        if file_path is None:
            differences.append(
                _difference(
                    DIFF_MODULE_FILE_MISSING,
                    name,
                    "file-backed module",
                    "no module file",
                    "module file recorded at serialization time cannot be " "located",
                )
            )
        else:
            actual_digest = _digest_file(file_path)
            if (
                actual_digest is None
                or actual_digest["digest"] != recorded_digest["digest"]
            ):
                differences.append(
                    _difference(
                        DIFF_FILE_DIGEST,
                        name,
                        recorded_digest["digest"],
                        actual_digest["digest"] if actual_digest else None,
                        "module file digest differs",
                    )
                )


def verify_manifest(data, *, policy="strict", allowed_differences=None):
    """Validate an envelope against the current environment.

    The validation only reads the manifest and the local environment: it
    does not reconstruct any pickle object and therefore cannot execute the
    payload's reconstructors.  All detected differences are collected and
    returned in a :class:`ManifestCheckReport`.

    Parameters
    ----------
    data:
        A framed envelope (bytes) or a :class:`ManifestEnvelope`.
    policy:
        Only ``"strict"`` is supported.
    allowed_differences:
        Optional iterable of difference kind constants (see
        :data:`MANIFEST_DIFFERENCE_KINDS`) that the caller explicitly
        accepts.
    """
    allowed = _normalize_policy(policy, allowed_differences)
    if isinstance(data, ManifestEnvelope):
        envelope = data
        ManifestEnvelope.validate_manifest_schema(envelope.manifest)
    else:
        envelope = ManifestEnvelope.from_bytes(data)

    manifest = envelope.manifest
    differences = []

    # 1. Payload integrity.  A mismatch means the bytes no longer correspond
    #    to what was serialized, so nothing else should be trusted.
    payload_section = manifest["payload"]
    try:
        actual_digest = hashlib.new(
            payload_section["algorithm"], envelope.payload
        ).hexdigest()
    except (ValueError, TypeError) as e:
        raise ManifestFormatError(
            "unknown digest algorithm {!r}".format(payload_section.get("algorithm"))
        ) from e
    if actual_digest != payload_section["digest"]:
        differences.append(
            _difference(
                DIFF_PAYLOAD_DIGEST,
                "payload",
                payload_section["digest"],
                actual_digest,
                "payload digest does not match the manifest: the payload may "
                "have been tampered with",
            )
        )

    # 2. Python implementation / version / bytecode ABI.
    recorded_python = manifest["python"]
    actual_implementation = platform.python_implementation()
    if recorded_python["implementation"] != actual_implementation:
        differences.append(
            _difference(
                DIFF_PYTHON_IMPLEMENTATION,
                "python.implementation",
                recorded_python["implementation"],
                actual_implementation,
                "Python implementation differs",
            )
        )

    recorded_version = tuple(recorded_python["version_info"])
    actual_version = tuple(sys.version_info)
    if recorded_version[:2] != actual_version[:2]:
        differences.append(
            _difference(
                DIFF_PYTHON_VERSION,
                "python.version_info",
                list(recorded_version),
                list(actual_version),
                "Python major/minor version differs",
            )
        )

    actual_cache_tag = sys.implementation.cache_tag
    if recorded_python["cache_tag"] != actual_cache_tag:
        differences.append(
            _difference(
                DIFF_CACHE_TAG,
                "python.cache_tag",
                recorded_python["cache_tag"],
                actual_cache_tag,
                "interpreter cache tag differs",
            )
        )

    actual_magic = int.from_bytes(importlib.util.MAGIC_NUMBER, "little")
    if (
        recorded_python.get("bytecode_dynamic")
        and recorded_python["magic_number"] != actual_magic
    ):
        differences.append(
            _difference(
                DIFF_BYTECODE_ABI,
                "python.magic_number",
                recorded_python["magic_number"],
                actual_magic,
                "bytecode magic differs while the payload carries code "
                "objects by value: CodeType reconstruction is ABI dependent",
            )
        )

    # 3. By-reference decisions and explicitly discovered submodules.
    for reference in manifest["references"]:
        _check_reference(differences, reference["module"], reference["qualname"])
    for submodule in manifest.get("submodules", []):
        _module, spec = _locate_module_without_executing(submodule)
        if _module is None and spec is None:
            differences.append(
                _difference(
                    DIFF_MODULE_MISSING,
                    submodule,
                    "importable submodule",
                    "not found",
                    "explicitly discovered submodule {!r} is not "
                    "importable".format(submodule),
                )
            )

    # 4. Module descriptors (kind / distribution / file digest).
    distribution_mapping = _LazyDistributionMapping()
    for name, entry in manifest["modules"].items():
        _check_module_entry(differences, name, entry, distribution_mapping)

    # 5. Opaque boundaries are surfaced as differences in strict mode: the
    #    caller must explicitly acknowledge them.
    for entry in manifest.get("opaque", []):
        subject = entry.get("reconstructor") or entry.get("object_type", "")
        differences.append(
            _difference(
                DIFF_OPAQUE_BOUNDARY,
                subject,
                "no opaque reducer/persistent boundary",
                entry,
                "payload contains an opaque {} boundary whose internal "
                "imports cannot be verified".format(entry["kind"]),
            )
        )

    return ManifestCheckReport(envelope, differences, allowed)


def loads_checked(data, *, policy="strict", allowed_differences=None, buffers=None):
    """Load an envelope after verifying its manifest against the environment.

    All checks (manifest format, payload digest, environment conditions) are
    performed before any pickle reconstruction starts; when they fail, a
    :class:`ManifestCheckError` carrying the structured difference list is
    raised and the payload is never handed to ``pickle``.

    A successful check only means that the caller's policy is satisfied: it
    does not turn an untrusted pickle into a trusted one, since unpickling
    still invokes reconstructor callables.  Nothing is installed or
    downloaded automatically.

    Parameters
    ----------
    data:
        A framed envelope (bytes) or a :class:`ManifestEnvelope`.
    policy:
        Only ``"strict"`` is supported.
    allowed_differences:
        Optional iterable of difference kind constants explicitly accepted
        by the caller.
    buffers:
        Optional iterable of out-of-band buffers, forwarded to
        :class:`pickle.Unpickler`.
    """
    report = verify_manifest(
        data, policy=policy, allowed_differences=allowed_differences
    )
    if not report.ok:
        raise ManifestCheckError(report)

    payload = report.envelope.payload
    if buffers is None:
        return pickle.loads(payload)
    with io.BytesIO(payload) as file:
        return pickle.Unpickler(file, buffers=buffers).load()
