"""Tests for the versioned dependency manifest / checked loading feature.

The feature under test lives in :mod:`cloudpickle.manifest` and exposes
``dumps_with_manifest``, ``loads_checked`` and ``verify_manifest``.

A cross-cutting concern of these tests is to prove that when the precheck
fails, *no* pickle reconstruction happens at all: the fixture package
exposes a reconstructor that counts every invocation (``reconstructor_writes``)
and every failing scenario asserts that the counter stays empty.
"""

import importlib
import json
import os
import pickle
import struct
import sys
import types

import pytest

import cloudpickle
from cloudpickle import (
    DIFF_BYTECODE_ABI,
    DIFF_CACHE_TAG,
    DIFF_DISTRIBUTION_VERSION,
    DIFF_FILE_DIGEST,
    DIFF_MODULE_KIND,
    DIFF_MODULE_MISSING,
    DIFF_OPAQUE_BOUNDARY,
    DIFF_PAYLOAD_DIGEST,
    DIFF_PYTHON_VERSION,
    DIFF_QUALNAME_MISSING,
    MANIFEST_VERSION,
    ManifestCheckError,
    ManifestEnvelope,
    ManifestFormatError,
    dumps_with_manifest,
    loads_checked,
    verify_manifest,
)

_FIXTURE_PKG_NAME = "cp_manifest_fixture_pkg"
_FIXTURE_DEP_NAME = "cp_manifest_dep_pkg"
_FIXTURE_NS_NAME = "cp_manifest_fixture_ns"
_MISSING_NAME = "cp_manifest_definitely_missing_module_xyz"

_FIXTURE_SOURCE = """
# Test fixture package for cloudpickle manifest tests.

reconstructor_writes = []
reduce_calls = 0
VERSION = "1.2.3"
SECRET_GLOBAL = "s3cr3t-manifest-global-value"


def counting_reconstructor(value):
    # Each invocation is recorded so that tests can prove that a failed
    # precheck never executes any reconstructor.
    reconstructor_writes.append(value)
    return Reconstructed(value + 1)


def make_counted(value):
    return Counted(value)


class Reconstructed:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, Reconstructed) and self.v == other.v

    def __reduce__(self):
        return (counting_reconstructor, (self.v,))


class Counted:
    def __init__(self, v):
        self.v = v

    def __reduce__(self):
        global reduce_calls
        reduce_calls += 1
        return (make_counted, (self.v,))


def add(a, b):
    return a + b


# Importing a dependency submodule at package initialization binds the
# dependency package in this module's globals and makes the submodule
# discoverable by cloudpickle's _find_imported_submodules.
import cp_manifest_dep_pkg.util


def uses_subpackage():
    # Attribute-style access to a submodule, which is the pattern that
    # cloudpickle._find_imported_submodules is designed to detect.
    return cp_manifest_dep_pkg.util.helper
"""

_SUBPACKAGE_SOURCE = """
def mod_function():
    return "from the subpackage"
"""

_DEP_UTIL_SOURCE = """
def helper():
    return "from the dependency submodule"
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixture_pkg(tmp_path, monkeypatch):
    """Importable file-backed package with a write-counting reconstructor."""
    pkg_dir = tmp_path / _FIXTURE_PKG_NAME
    sub_dir = pkg_dir / "sub"
    sub_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(_FIXTURE_SOURCE, encoding="utf-8")
    (sub_dir / "__init__.py").write_text(_SUBPACKAGE_SOURCE, encoding="utf-8")

    dep_dir = tmp_path / _FIXTURE_DEP_NAME
    dep_util_dir = dep_dir / "util"
    dep_util_dir.mkdir(parents=True)
    (dep_dir / "__init__.py").write_text("", encoding="utf-8")
    (dep_util_dir / "__init__.py").write_text(_DEP_UTIL_SOURCE, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    pkg = importlib.import_module(_FIXTURE_PKG_NAME)
    yield pkg
    sys.modules.pop(_FIXTURE_PKG_NAME, None)
    sys.modules.pop(_FIXTURE_PKG_NAME + ".sub", None)
    sys.modules.pop(_FIXTURE_DEP_NAME, None)
    sys.modules.pop(_FIXTURE_DEP_NAME + ".util", None)


@pytest.fixture
def namespace_pkg(tmp_path, monkeypatch):
    """PEP 420 namespace package: a directory without ``__init__.py``."""
    ns_dir = tmp_path / _FIXTURE_NS_NAME
    ns_dir.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    pkg = importlib.import_module(_FIXTURE_NS_NAME)
    yield pkg
    sys.modules.pop(_FIXTURE_NS_NAME, None)


def _tampered_manifest(envelope, mutations):
    manifest = json.loads(json.dumps(envelope.manifest))
    for path, value in mutations:
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return ManifestEnvelope(manifest, envelope.payload).to_bytes()


def _blocking_kinds(error):
    return [d["kind"] for d in error.report.blocking]


def _all_kinds(report):
    return [d["kind"] for d in report.differences]


# ---------------------------------------------------------------------------
# Envelope framing and basic roundtrip
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_roundtrip_and_determinism(self, fixture_pkg):
        obj = {"func": fixture_pkg.add, "values": [1, 2, 3]}
        envelope = dumps_with_manifest(obj)
        raw = envelope.to_bytes()

        assert raw[:8] == b"CPCKMNF1"
        parsed = ManifestEnvelope.from_bytes(raw)
        assert parsed.manifest == envelope.manifest
        assert parsed.payload == envelope.payload
        # Framing must be stable and deterministic.
        assert parsed.to_bytes() == raw
        assert dumps_with_manifest(obj).to_bytes() == raw

    @pytest.mark.parametrize("protocol", [0, 1, 2, 4, 5, pickle.HIGHEST_PROTOCOL])
    def test_roundtrip_all_protocols(self, fixture_pkg, protocol):
        envelope = dumps_with_manifest(fixture_pkg.add, protocol=protocol)
        report = verify_manifest(envelope)
        assert report.ok
        assert loads_checked(envelope) is fixture_pkg.add

    def test_bytes_input_accepted(self, fixture_pkg):
        raw = dumps_with_manifest(fixture_pkg.add).to_bytes()
        assert loads_checked(raw) is fixture_pkg.add

    def test_format_errors_never_reconstruct(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(1))
        raw = envelope.to_bytes()

        bad_cases = [
            b"not-a-manifest",
            raw[:10],  # truncated
            b"CPCKMNF1" + struct.pack(">Q", 10**9) + b"{}",
        ]
        for bad in bad_cases:
            fixture_pkg.reconstructor_writes.clear()
            with pytest.raises(ManifestFormatError):
                loads_checked(bad)
            assert fixture_pkg.reconstructor_writes == []

        # Well framed but unparseable JSON.
        body = b"{not json"
        bad_json = b"CPCKMNF1" + struct.pack(">Q", len(body)) + body + envelope.payload
        with pytest.raises(ManifestFormatError):
            ManifestEnvelope.from_bytes(bad_json)

        # Valid JSON but unsupported schema.
        with pytest.raises(ManifestFormatError):
            ManifestEnvelope.from_bytes(
                _tampered_manifest(envelope, [(("manifest_version",), 999)])
            )
        with pytest.raises(ManifestFormatError):
            ManifestEnvelope.from_bytes(
                _tampered_manifest(envelope, [(("references",), "not-a-list")])
            )

    def test_manifest_version_constant(self):
        envelope = dumps_with_manifest(1)
        assert envelope.manifest["manifest_version"] == MANIFEST_VERSION == 1


# ---------------------------------------------------------------------------
# Manifest content: actual reducer decisions
# ---------------------------------------------------------------------------


class TestManifestContent:
    def test_by_reference_module_and_qualname(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        refs = {(r["module"], r["qualname"]) for r in envelope.manifest["references"]}
        assert (_FIXTURE_PKG_NAME, "add") in refs
        # The same decisions are exposed as a module entry.
        assert _FIXTURE_PKG_NAME in envelope.manifest["modules"]

    def test_module_kinds(self, fixture_pkg, namespace_pkg):
        dynamic = types.ModuleType("cp_manifest_runtime_dynamic_module")
        dynamic.value = 7
        sys.modules[dynamic.__name__] = dynamic
        try:
            envelope = dumps_with_manifest(
                {
                    "builtin": sys,
                    "file_module": fixture_pkg,
                    "namespace": namespace_pkg,
                    "dynamic": dynamic,
                }
            )
        finally:
            sys.modules.pop(dynamic.__name__, None)

        modules = envelope.manifest["modules"]
        assert modules["sys"]["kind"] == "builtin"
        assert modules[_FIXTURE_PKG_NAME]["kind"] == "package"
        assert modules[_FIXTURE_NS_NAME]["kind"] == "namespace"
        # A module registered in sys.modules without an import spec is a
        # fileless dynamic module; it is referenced by name but still labeled
        # correctly from introspection.
        assert modules["cp_manifest_runtime_dynamic_module"]["kind"] == "dynamic"

        # Dynamic / unobserved entries never carry file information.
        dynamic_entry = modules["cp_manifest_runtime_dynamic_module"]
        assert dynamic_entry["file_digest"] is None

    def test_fileless_dynamic_module_not_in_sys_modules(self):
        dynamic = types.ModuleType("cp_manifest_orphan_dynamic_module")
        dynamic.value = 1
        envelope = dumps_with_manifest(dynamic)
        entry = envelope.manifest["modules"]["cp_manifest_orphan_dynamic_module"]
        # The dynamic_subimport reducer decision is authoritative even when
        # the module never reached sys.modules.
        assert entry["kind"] == "dynamic"
        restored = loads_checked(envelope)
        assert restored.value == 1

    def test_dynamic_by_value_function_carries_code_abi(self):
        def local_function(x):
            return x * 2

        envelope = dumps_with_manifest(local_function)
        dynamic = envelope.manifest["dynamic"]
        assert dynamic["bytecode_dynamic"] is True
        assert dynamic["code_object_count"] >= 1
        assert dynamic["dynamic_function_count"] == 1
        python_section = envelope.manifest["python"]
        assert python_section["implementation"] == "CPython"
        assert python_section["cache_tag"] == sys.implementation.cache_tag
        assert isinstance(python_section["magic_number"], int)
        assert loads_checked(envelope)(21) == 42

    def test_no_code_abi_flag_for_reference_only_payload(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        assert envelope.manifest["dynamic"]["bytecode_dynamic"] is False
        assert envelope.manifest["dynamic"]["code_object_count"] == 0

    def test_explicit_submodule_discovery(self, fixture_pkg):
        import cp_manifest_dep_pkg

        cloudpickle.register_pickle_by_value(fixture_pkg)
        try:
            envelope = dumps_with_manifest(fixture_pkg.uses_subpackage)
        finally:
            cloudpickle.unregister_pickle_by_value(fixture_pkg)
        submodules = envelope.manifest["submodules"]
        assert _FIXTURE_DEP_NAME + ".util" in submodules
        # The discovered submodule is also described as a regular module.
        assert _FIXTURE_DEP_NAME + ".util" in envelope.manifest["modules"]
        assert loads_checked(envelope)() is cp_manifest_dep_pkg.util.helper

    def test_manifest_never_contains_globals_or_paths(self, fixture_pkg):
        cloudpickle.register_pickle_by_value(fixture_pkg)
        try:
            envelope = dumps_with_manifest(
                fixture_pkg.uses_subpackage, include_file_digests=True
            )
        finally:
            cloudpickle.unregister_pickle_by_value(fixture_pkg)
        raw_manifest = json.dumps(envelope.manifest)
        # No module global value leaks into the manifest.
        assert fixture_pkg.SECRET_GLOBAL not in raw_manifest
        # No file system path leaks: only non-reversible digests are kept.
        assert fixture_pkg.__file__ not in raw_manifest
        assert os.path.dirname(fixture_pkg.__file__) not in raw_manifest
        digest_entries = [
            e["file_digest"]
            for e in envelope.manifest["modules"].values()
            if e["file_digest"]
        ]
        assert digest_entries and all(
            set(e) == {"algorithm", "digest"} for e in digest_entries
        )

    def test_no_speculative_imports_at_dump_time(
        self, tmp_path, monkeypatch, fixture_pkg
    ):
        # A package directory exists on sys.path but is never imported: the
        # dump side must not import a candidate module as a probe. Distribution
        # metadata is disabled because its first use lazily imports unrelated
        # stdlib parsers, which is metadata machinery rather than module
        # probing.
        candidate_dir = tmp_path / _MISSING_NAME
        candidate_dir.mkdir()
        (candidate_dir / "__init__.py").write_text("x = 1\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        before = set(sys.modules)
        dumps_with_manifest(
            {"add": fixture_pkg.add, "path_join": os.path.join},
            include_distributions=False,
            include_file_digests=False,
        )
        after = set(sys.modules)
        assert after == before
        assert _MISSING_NAME not in sys.modules

        # Even with metadata enabled, the candidate itself stays untouched.
        dumps_with_manifest(fixture_pkg.add, include_distributions=True)
        assert _MISSING_NAME not in sys.modules

    def test_reducer_not_called_more_than_plain_dumps(self, fixture_pkg):
        obj = fixture_pkg.Counted(42)
        fixture_pkg.reduce_calls = 0
        cloudpickle.dumps(obj)
        plain_calls = fixture_pkg.reduce_calls

        fixture_pkg.reduce_calls = 0
        dumps_with_manifest(obj)
        manifest_calls = fixture_pkg.reduce_calls

        assert plain_calls == manifest_calls == 1

        # Memoized duplicates are reduced once in both cases.
        shared = fixture_pkg.Counted(7)
        fixture_pkg.reduce_calls = 0
        cloudpickle.dumps([shared, shared])
        plain_calls = fixture_pkg.reduce_calls
        fixture_pkg.reduce_calls = 0
        dumps_with_manifest([shared, shared])
        assert fixture_pkg.reduce_calls == plain_calls == 1

    def test_distribution_metadata_is_optional_and_best_effort(self):
        # A dynamic function always references cloudpickle itself, which has
        # distribution metadata in most environments.
        def f():
            return None

        envelope = dumps_with_manifest(f, include_distributions=True)
        entry = envelope.manifest["modules"]["cloudpickle.cloudpickle"]
        if entry["distribution"] is not None:
            assert set(entry["distribution"]) == {"name", "version"}
        no_dist = dumps_with_manifest(f, include_distributions=False)
        assert all(
            e["distribution"] is None for e in no_dist.manifest["modules"].values()
        )


# ---------------------------------------------------------------------------
# Opaque boundaries
# ---------------------------------------------------------------------------


class TestOpaqueBoundaries:
    def test_external_reconstructor_flagged(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(10))
        opaque = envelope.manifest["opaque"]
        assert len(opaque) == 1
        assert opaque[0]["kind"] == "external_reconstructor"
        assert opaque[0]["reconstructor"] == (
            _FIXTURE_PKG_NAME + ".counting_reconstructor"
        )

    def test_strict_opaque_failure_executes_nothing(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(10))
        fixture_pkg.reconstructor_writes.clear()
        report = verify_manifest(envelope)
        assert not report.ok
        assert _all_kinds(report) == [DIFF_OPAQUE_BOUNDARY]
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(envelope)
        assert DIFF_OPAQUE_BOUNDARY in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_allowed_opaque_loads_like_plain_loads(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(10))
        fixture_pkg.reconstructor_writes.clear()
        result = loads_checked(envelope, allowed_differences=[DIFF_OPAQUE_BOUNDARY])
        # Exactly one reconstruction write, and parity with plain loads.
        assert fixture_pkg.reconstructor_writes == [10]
        assert result == pickle.loads(envelope.payload)

    def test_dynamic_reconstructor_flagged(self, fixture_pkg):
        def build_holder():
            def local_reconstructor(v):
                return fixture_pkg.Reconstructed(v)

            class Holder:
                def __init__(self, v):
                    self.v = v

                def __reduce__(self):
                    # A reconstructor carried by value: its internal imports
                    # cannot be discovered from the outside.
                    return (local_reconstructor, (self.v,))

            return Holder

        Holder = build_holder()
        envelope = dumps_with_manifest(Holder(3))
        kinds = {entry["kind"] for entry in envelope.manifest["opaque"]}
        assert "dynamic_reconstructor" in kinds

        fixture_pkg.reconstructor_writes.clear()
        with pytest.raises(ManifestCheckError):
            loads_checked(envelope)
        assert fixture_pkg.reconstructor_writes == []
        loaded = loads_checked(envelope, allowed_differences=[DIFF_OPAQUE_BOUNDARY])
        assert isinstance(loaded, Holder) or loaded.v == 3

    def test_persistent_id_boundary(self, fixture_pkg):
        class Persistent:
            pass

        def persistent_id(obj):
            return "persistent-flag" if isinstance(obj, Persistent) else None

        envelope = dumps_with_manifest(Persistent(), persistent_id=persistent_id)
        opaque = envelope.manifest["opaque"]
        assert any(entry["kind"] == "persistent_id" for entry in opaque)
        # Only the type name is recorded, never the id payload itself.
        assert "persistent-flag" not in json.dumps(envelope.manifest)


# ---------------------------------------------------------------------------
# Precheck failures: zero reconstruction execution
# ---------------------------------------------------------------------------


class TestPrecheckFailures:
    def _envelope_with_counter(self, fixture_pkg):
        return dumps_with_manifest(fixture_pkg.Reconstructed(10))

    def test_payload_tampering(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        raw = bytearray(envelope.to_bytes())
        raw[-1] ^= 0x01  # flip a payload byte
        fixture_pkg.reconstructor_writes.clear()
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(bytes(raw))
        assert DIFF_PAYLOAD_DIGEST in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_missing_module_reference(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        framed = _tampered_manifest(
            envelope,
            [
                (
                    ("references",),
                    envelope.manifest["references"]
                    + [{"module": _MISSING_NAME, "qualname": "anything"}],
                )
            ],
        )
        fixture_pkg.reconstructor_writes.clear()
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(framed)
        assert DIFF_MODULE_MISSING in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_missing_submodule(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        framed = _tampered_manifest(
            envelope,
            [(("submodules",), [_MISSING_NAME + ".sub"])],
        )
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(framed)
        assert DIFF_MODULE_MISSING in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_qualname_missing(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        removed = fixture_pkg.add
        try:
            delattr(fixture_pkg, "add")
            report = verify_manifest(envelope)
        finally:
            fixture_pkg.add = removed
        assert DIFF_QUALNAME_MISSING in _all_kinds(report)

    def test_python_version_mismatch(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        framed = _tampered_manifest(
            envelope,
            [(("python", "version_info"), [3, 99, 0, "final", 0])],
        )
        fixture_pkg.reconstructor_writes.clear()
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(framed)
        assert DIFF_PYTHON_VERSION in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_bytecode_abi_mismatch_for_dynamic_code(self):
        def local_function():
            return 1

        envelope = dumps_with_manifest(local_function)
        magic = envelope.manifest["python"]["magic_number"]
        framed = _tampered_manifest(
            envelope,
            [(("python", "magic_number"), magic + 1)],
        )
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(framed)
        assert DIFF_BYTECODE_ABI in _blocking_kinds(exc_info.value)

    def test_cache_tag_mismatch(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        framed = _tampered_manifest(envelope, [(("python", "cache_tag"), "pypy-999")])
        with pytest.raises(ManifestCheckError) as exc_info:
            loads_checked(framed)
        assert DIFF_CACHE_TAG in _blocking_kinds(exc_info.value)
        assert fixture_pkg.reconstructor_writes == []

    def test_file_digest_mismatch_after_module_change(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add, include_file_digests=True)
        module_path = fixture_pkg.__file__
        original = open(module_path, encoding="utf-8").read()
        try:
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(original + "\n# changed after serialization\n")
            report = verify_manifest(envelope)
            assert DIFF_FILE_DIGEST in _all_kinds(report)
            with pytest.raises(ManifestCheckError):
                loads_checked(envelope)
            # Explicitly accepting the digest difference still loads.
            loaded = loads_checked(envelope, allowed_differences=[DIFF_FILE_DIGEST])
            assert loaded is fixture_pkg.add
        finally:
            with open(module_path, "w", encoding="utf-8") as f:
                f.write(original)

    def test_distribution_version_mismatch(self):
        def local_function():
            return None

        envelope = dumps_with_manifest(local_function)
        target = envelope.manifest["modules"]["cloudpickle.cloudpickle"]
        if target["distribution"] is None:
            pytest.skip("cloudpickle distribution metadata is unavailable")
        framed = _tampered_manifest(
            envelope,
            [
                (
                    ("modules", "cloudpickle.cloudpickle", "distribution", "version"),
                    "99.0.0",
                )
            ],
        )
        report = verify_manifest(ManifestEnvelope.from_bytes(framed))
        assert DIFF_DISTRIBUTION_VERSION in _all_kinds(report)
        diff = next(
            d for d in report.differences if d["kind"] == DIFF_DISTRIBUTION_VERSION
        )
        assert diff["expected"] == "99.0.0"
        assert diff["actual"] == target["distribution"]["version"]

    def test_module_kind_mismatch(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        framed = _tampered_manifest(
            envelope,
            [(("modules", _FIXTURE_PKG_NAME, "kind"), "builtin")],
        )
        report = verify_manifest(ManifestEnvelope.from_bytes(framed))
        assert DIFF_MODULE_KIND in _all_kinds(report)

    def test_all_differences_collected_structured(self, fixture_pkg):
        envelope = self._envelope_with_counter(fixture_pkg)
        framed = _tampered_manifest(
            envelope,
            [
                (("python", "version_info"), [3, 99, 0, "final", 0]),
                (
                    ("references",),
                    envelope.manifest["references"]
                    + [{"module": _MISSING_NAME, "qualname": "x"}],
                ),
                (
                    ("python", "magic_number"),
                    envelope.manifest["python"]["magic_number"] + 1,
                ),
            ],
        )
        # Also tamper the payload itself.
        raw = bytearray(framed)
        raw[-1] ^= 0x01
        report = verify_manifest(bytes(raw))
        kinds = set(_all_kinds(report))
        assert {
            DIFF_PAYLOAD_DIGEST,
            DIFF_PYTHON_VERSION,
            DIFF_MODULE_MISSING,
        }.issubset(kinds)
        for difference in report.differences:
            assert set(["kind", "subject", "expected", "actual", "message"]).issubset(
                difference
            )
        assert fixture_pkg.reconstructor_writes == []


# ---------------------------------------------------------------------------
# Policy handling
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_unknown_policy_rejected(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        with pytest.raises(ValueError):
            loads_checked(envelope, policy="anything-goes")

    def test_unknown_allowed_kind_rejected(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        with pytest.raises(ValueError):
            loads_checked(envelope, allowed_differences=["not-a-real-kind"])

    def test_allowed_difference_makes_report_ok(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.add)
        framed = _tampered_manifest(
            envelope,
            [(("modules", _FIXTURE_PKG_NAME, "kind"), "builtin")],
        )
        report = verify_manifest(ManifestEnvelope.from_bytes(framed))
        assert not report.ok
        relaxed = verify_manifest(
            ManifestEnvelope.from_bytes(framed),
            allowed_differences=[DIFF_MODULE_KIND],
        )
        assert relaxed.ok
        assert (
            loads_checked(framed, allowed_differences=[DIFF_MODULE_KIND])
            is fixture_pkg.add
        )

    def test_payload_digest_cannot_be_allowed_to_heal_tampering(self, fixture_pkg):
        # Allowing the digest kind only suppresses the policy error: it does
        # not turn corrupted bytes into a valid object. The precheck passes
        # but plain pickle then reports the stream problem itself.
        envelope = dumps_with_manifest(fixture_pkg.add)
        raw = bytearray(envelope.to_bytes())
        raw[-1] ^= 0x01
        report = verify_manifest(bytes(raw), allowed_differences=[DIFF_PAYLOAD_DIGEST])
        assert report.ok
        with pytest.raises(Exception):
            loads_checked(bytes(raw), allowed_differences=[DIFF_PAYLOAD_DIGEST])


# ---------------------------------------------------------------------------
# Success path parity with plain loads
# ---------------------------------------------------------------------------


class TestSuccessParity:
    def test_checked_load_equals_plain_loads(self, fixture_pkg):
        cases = [
            fixture_pkg.add,
            [1, "a", None, fixture_pkg.add],
            {"nested": {"f": fixture_pkg.add}},
        ]
        for obj in cases:
            envelope = dumps_with_manifest(obj)
            assert loads_checked(envelope) == pickle.loads(envelope.payload)

    def test_dynamic_function_parity(self):
        def make_closure(n):
            factor = n

            def inner(x):
                return x * factor

            return inner

        func = make_closure(3)
        envelope = dumps_with_manifest(func)
        assert loads_checked(envelope)(7) == pickle.loads(envelope.payload)(7) == 21

    def test_verify_does_not_reconstruct(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(1))
        fixture_pkg.reconstructor_writes.clear()
        report = verify_manifest(envelope, allowed_differences=[DIFF_OPAQUE_BOUNDARY])
        assert report.ok
        assert fixture_pkg.reconstructor_writes == []

    def test_write_counter_one_increment_on_success(self, fixture_pkg):
        envelope = dumps_with_manifest(fixture_pkg.Reconstructed(11))
        fixture_pkg.reconstructor_writes.clear()
        loads_checked(envelope, allowed_differences=[DIFF_OPAQUE_BOUNDARY])
        assert fixture_pkg.reconstructor_writes == [11]
        # A second checked load is a second genuine unpickling, not a cached
        # precheck artifact.
        loads_checked(envelope, allowed_differences=[DIFF_OPAQUE_BOUNDARY])
        assert fixture_pkg.reconstructor_writes == [11, 11]
