"""Versioned length-framed bundles for pickle protocol 5 out-of-band buffers.

This module provides :func:`dump_bundle` and :func:`load_bundle`, a container
format that wraps the two outputs of a protocol 5 pickling run:

* the in-band pickle payload (the bytes produced by the pickler), and
* the ordered sequence of :class:`pickle.PickleBuffer` objects handed to a
  ``buffer_callback``.

It does **not** change which objects are serialized out-of-band: that decision
is entirely left to the pickler and to the reducers of the serialized objects.
The bundle layer only adds:

* explicit, versioned length framing (all multi-byte integers are stored as
  little-endian and the byte order is recorded in the header, so no equality of
  endianness between the producer and the consumer is ever assumed);
* an integrity manifest, written last, recording the order, length,
  read-only flag and digest of every buffer, the digest of the payload and the
  total lengths;
* resource limits enforced *before* data is read, using bounded chunked I/O;
* full validation of every frame boundary, digest, count and limit before any
  pickle reconstruction is attempted.

Repeated references to the same :class:`~pickle.PickleBuffer` are emitted by the
pickler as repeated callback entries. They are stored as explicit *alias*
frames sharing a single backing storage, and reconstructed as repeated entries
of the very same buffer object, exactly like calling
``pickle.loads(payload, buffers=buffers)`` with the original callback list.

Security note
-------------
Digests and framing only detect accidental corruption or tampering; they do
**not** make unpickling safe. Loading a bundle still executes the reconstructors
embedded in the pickle payload, so bundles must never be loaded from sources
that are not already trusted, exactly like regular pickle streams.

Ownership model
---------------
:func:`load_bundle` returns a :class:`LoadedBundle`, which owns every backing
resource created for the reconstructed buffers (in-memory byte strings or
memory-mapped temporary files). The reconstructed object graph may hold views
over that storage, so the bundle must be kept alive as long as the
reconstructed object is in use. Closing the bundle while such views are still
exported fails explicitly with a :class:`BundleError` instead of leaving
dangling buffers.
"""

import hashlib
import hmac
import io
import mmap
import os
import pickle
import pickletools
import platform
import struct
import tempfile

from .cloudpickle import DEFAULT_PROTOCOL, Pickler

# ----------------------------------------------------------------------------
# Format constants
# ----------------------------------------------------------------------------

BUNDLE_FORMAT_VERSION = 1

_MAGIC = b"CPKLBNDL"
_COMMIT_MAGIC = b"CPKDONE!\n"
_ENDIAN_LE = ord("<")
_DIGEST_SHA256 = 1
_DIGEST_SIZE = 32
_DIGEST_NAME = "sha256"

_FRAME_PAYLOAD = 1
_FRAME_BUFFER = 2
_FRAME_ALIAS = 3
_FRAME_MANIFEST = 4

_FLAG_READONLY = 0x01
_KNOWN_FLAGS = _FLAG_READONLY

_MANIFEST_KIND_DATA = 1
_MANIFEST_KIND_ALIAS = 2

# Default I/O chunk size: individual read()/write() calls never exceed this.
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB

# Per-buffer threshold above which a loaded buffer is spilled to a temporary
# memory-mapped file instead of being held in memory.
DEFAULT_SPOOL_SIZE = 8 << 20  # 8 MiB

# Default resource limits enforced on both writing and reading.
DEFAULT_MAX_FRAME_LENGTH = 1 << 30  # 1 GiB per single frame
DEFAULT_MAX_TOTAL_LENGTH = 8 << 30  # 8 GiB of payload + buffer data
DEFAULT_MAX_BUFFERS = 100_000

_U8 = struct.Struct("<B")
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------


class BundleError(Exception):
    """Base class for all bundle format errors."""


class BundleFormatError(BundleError):
    """Raised when a bundle does not conform to the container format.

    This includes bad magic, unsupported versions or endianness, invalid frame
    boundaries, unexpected end of stream (short reads) or trailing bytes.
    """


class BundleIntegrityError(BundleError):
    """Raised when a digest check or a manifest cross-check fails.

    This also covers missing or extra buffers with respect to the pickle
    payload and the committed manifest.
    """


class BundleResourceLimit(BundleError):
    """Raised when a declared length or count exceeds the configured limits."""


class BundleCancelled(BundleError):
    """Raised when a ``should_cancel`` callback requests cancellation."""


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _check_positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _read_exact(stream, size):
    """Read exactly ``size`` bytes from ``stream`` or raise on short read.

    The read is performed in bounded chunks so that a huge declared length can
    never translate into a single unbounded allocation.
    """
    chunks = []
    remaining = size
    while remaining:
        block = stream.read(min(remaining, DEFAULT_CHUNK_SIZE))
        if not block:
            raise BundleFormatError(
                f"unexpected end of stream: {size - remaining} of {size} " "bytes read"
            )
        if not isinstance(block, (bytes, bytearray, memoryview)):
            raise BundleFormatError(
                f"stream.read() returned non-bytes object of type "
                f"{type(block).__name__}"
            )
        chunks.append(block)
        remaining -= len(block)
    if len(chunks) == 1:
        return chunks[0]
    return b"".join(chunks)


def _iter_view_chunks(view, chunk_size):
    """Yield memoryview slices of ``view`` without copying the data."""
    for start in range(0, view.nbytes, chunk_size):
        yield view[start : start + chunk_size]


def _digest_view(view, chunk_size):
    hasher = hashlib.sha256()
    for chunk in _iter_view_chunks(view, chunk_size):
        hasher.update(chunk)
    return hasher.digest()


# ----------------------------------------------------------------------------
# Spooled per-frame storage owned by the loader
# ----------------------------------------------------------------------------


class _SpooledSink:
    """Chunked byte sink kept in memory, then spilled to a temporary file.

    One sink is created per logical frame, so the contents of distinct buffers
    are never concatenated into a single aggregate bytes object. The sink owns
    its backing storage explicitly; a rolled-over temporary file is deleted by
    the operating system when it is closed.
    """

    def __init__(self, spool_size):
        self._spool_size = spool_size
        self._file = io.BytesIO()
        self._rolled = False
        self.length = 0

    def write(self, data):
        if not self._rolled and self.length + len(data) > self._spool_size:
            rolled_file = tempfile.TemporaryFile()
            previous = self._file
            previous.seek(0)
            while True:
                block = previous.read(DEFAULT_CHUNK_SIZE)
                if not block:
                    break
                rolled_file.write(block)
            self._file = rolled_file
            self._rolled = True
            previous.close()
        self._file.write(data)
        self.length += len(data)

    @property
    def rolled(self):
        """Whether the sink spilled its contents to a temporary file."""
        return self._rolled

    def getvalue(self):
        """Return the in-memory contents; only valid before rollover."""
        if self._rolled:
            raise BundleError("sink has spilled to disk and has no getvalue()")
        return self._file.getvalue()

    def seek(self, *args):
        return self._file.seek(*args)

    def read(self, *args):
        return self._file.read(*args)

    def readline(self):
        return self._file.readline()

    def materialize(self, readonly):
        """Return a buffer-provider backing object for a PickleBuffer.

        In-memory sinks become ``bytes`` (read-only) or ``bytearray``
        (writable); spilled sinks become a read-only or writable
        :class:`mmap.mmap`. Zero-length buffers never use mmap, which cannot
        map an empty file.
        """
        if not self._rolled:
            raw = self._file.getvalue()
            if readonly:
                return raw
            return bytearray(raw)
        self._file.flush()
        if self.length == 0:
            return b"" if readonly else bytearray()
        access = mmap.ACCESS_READ if readonly else mmap.ACCESS_WRITE
        return mmap.mmap(self._file.fileno(), self.length, access=access)

    def close(self):
        self._file.close()


# ----------------------------------------------------------------------------
# Public manifest and loaded bundle
# ----------------------------------------------------------------------------


class BundleManifest:
    """Read-only description of the contents of a validated bundle."""

    __slots__ = (
        "alias_count",
        "data_frame_count",
        "digest_name",
        "entries",
        "format_version",
        "payload_digest",
        "payload_length",
        "pickle_protocol",
        "producer",
        "slot_count",
        "total_buffer_length",
    )

    def __init__(self, **kwargs):
        for slot in self.__slots__:
            setattr(self, slot, kwargs[slot])

    def __repr__(self):
        return (
            f"BundleManifest(format_version={self.format_version}, "
            f"pickle_protocol={self.pickle_protocol}, "
            f"digest_name={self.digest_name!r}, producer={self.producer!r}, "
            f"slot_count={self.slot_count}, "
            f"data_frame_count={self.data_frame_count}, "
            f"alias_count={self.alias_count}, "
            f"payload_length={self.payload_length}, "
            f"total_buffer_length={self.total_buffer_length})"
        )


class LoadedBundle:
    """Handle owning the backing storage of a loaded bundle.

    Attributes
    ----------
    object : object
        The reconstructed object hierarchy.
    buffers : tuple of pickle.PickleBuffer
        The buffer slots, in pickle callback order. Alias slots repeat the
        same :class:`~pickle.PickleBuffer` object as their target slot.
    manifest : BundleManifest
        The validated manifest.

    The bundle is a context manager. It must remain open (and therefore keep
    its memory-mapped temporary files alive) as long as the reconstructed
    object may reference its buffers.
    """

    def __init__(self, obj, buffers, manifest, resources):
        self.object = obj
        self.buffers = tuple(buffers)
        self.manifest = manifest
        # resources is a tuple of (pickle_buffer, backing, sink, is_mmap)
        self._resources = resources
        self._closed = False

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Release every PickleBuffer and backing resource owned here.

        Raises :class:`BundleError` if reconstructed objects still export
        views over memory-mapped storage: closing such a mapping would
        otherwise leave the views dangling.
        """
        if self._closed:
            return
        # First release the PickleBuffer wrappers. Outstanding views exported
        # by application objects keep the underlying buffer alive on their
        # own until they are themselves released.
        for pickle_buffer, _backing, _sink, _is_mmap in self._resources:
            try:
                pickle_buffer.release()
            except ValueError:
                # Already released: nothing to do.
                pass
        for _pickle_buffer, backing, _sink, is_mmap in self._resources:
            if is_mmap:
                try:
                    backing.close()
                except BufferError as exc:
                    raise BundleError(
                        "cannot close bundle: the reconstructed object graph "
                        "still exports views over this bundle's memory-mapped "
                        "storage. Release those views (and the reconstructed "
                        "objects holding them) before closing the bundle."
                    ) from exc
        for _pickle_buffer, _backing, sink, _is_mmap in self._resources:
            sink.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f"<LoadedBundle closed={self._closed} {self.manifest!r}>"


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------


def _open_sink(file, mode):
    if isinstance(file, (str, bytes, os.PathLike)):
        return open(file, mode), True
    return file, False


def dump_bundle(
    obj,
    file,
    *,
    protocol=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_frame_length=DEFAULT_MAX_FRAME_LENGTH,
    max_total_length=DEFAULT_MAX_TOTAL_LENGTH,
    max_buffers=DEFAULT_MAX_BUFFERS,
):
    """Serialize ``obj`` and its protocol 5 buffers as a versioned bundle.

    Parameters
    ----------
    obj : object
        The object to serialize, using :class:`cloudpickle.Pickler`.
    file : path-like or binary writable file-like object
        Destination of the bundle. A file-like object is never closed by this
        function; a path is opened and closed internally.
    protocol : int, optional
        Pickle protocol, defaulting to :data:`cloudpickle.DEFAULT_PROTOCOL`.
        Protocol 5 or later is required for out-of-band buffers.
    chunk_size : int
        Maximum size of an individual write; buffer data is streamed as
        memoryview slices and is never concatenated into an aggregate copy.
    max_frame_length, max_total_length, max_buffers : int
        Resource limits also recorded in the bundle header. They are enforced
        while writing, so an oversized object fails before any partial output
        is committed.

    The completion manifest is the very last thing written. Any error aborts
    the bundle without a manifest, and internally opened destination files are
    removed; caller-provided streams are left open.
    """
    _check_positive_int("chunk_size", chunk_size)
    _check_positive_int("max_frame_length", max_frame_length)
    _check_positive_int("max_total_length", max_total_length)
    _check_positive_int("max_buffers", max_buffers)
    if protocol is None:
        protocol = DEFAULT_PROTOCOL
    protocol = int(protocol)
    if protocol < 5:
        raise ValueError(
            "dump_bundle requires pickle protocol >= 5 to carry out-of-band "
            f"buffers, got protocol {protocol}"
        )

    # Pickle first, into an in-memory payload that only contains pickle
    # metadata (no out-of-band buffer contents).
    payload_file = io.BytesIO()
    pickle_buffers = []
    Pickler(
        payload_file,
        protocol=protocol,
        buffer_callback=pickle_buffers.append,
    ).dump(obj)
    payload = payload_file.getvalue()
    del payload_file

    if len(payload) > max_frame_length:
        raise BundleResourceLimit(
            f"pickle payload is {len(payload)} bytes long, exceeding "
            f"max_frame_length={max_frame_length}"
        )
    if len(payload) > max_total_length:
        raise BundleResourceLimit(
            f"pickle payload is {len(payload)} bytes long, exceeding "
            f"max_total_length={max_total_length}"
        )
    if len(pickle_buffers) > max_buffers:
        raise BundleResourceLimit(
            f"object produced {len(pickle_buffers)} buffer slots, exceeding "
            f"max_buffers={max_buffers}"
        )

    from . import __version__

    stream, owned = _open_sink(file, "wb")
    committed = False
    try:
        producer = (
            f"cloudpickle {__version__} ; "
            f"{platform.python_implementation()} {platform.python_version()}"
        ).encode()
        header = b"".join(
            (
                _MAGIC,
                _U8.pack(BUNDLE_FORMAT_VERSION),
                _U8.pack(_ENDIAN_LE),
                _U8.pack(protocol),
                _U8.pack(_DIGEST_SHA256),
                _U8.pack(0),  # reserved flags
                _U16.pack(0),  # reserved
                _U32.pack(len(producer)),
                producer,
                _U64.pack(max_frame_length),
                _U64.pack(max_total_length),
            )
        )
        stream.write(header)

        # Payload frame.
        with memoryview(payload) as payload_view:
            payload_digest = _digest_view(payload_view, chunk_size)
            stream.write(_U8.pack(_FRAME_PAYLOAD))
            stream.write(_U64.pack(payload_view.nbytes))
            stream.write(payload_digest)
            for chunk in _iter_view_chunks(payload_view, chunk_size):
                stream.write(chunk)

        # Buffer frames, one per pickle callback slot. Repeated references to
        # the same PickleBuffer object become alias frames sharing storage.
        first_slot_by_id = {}
        manifest_entries = []
        total_buffer_length = 0
        for slot_index, pickle_buffer in enumerate(pickle_buffers):
            buffer_id = id(pickle_buffer)
            if buffer_id in first_slot_by_id:
                target = first_slot_by_id[buffer_id]
                stream.write(_U8.pack(_FRAME_ALIAS))
                stream.write(_U32.pack(slot_index))
                stream.write(_U32.pack(target))
                manifest_entries.append((_MANIFEST_KIND_ALIAS, slot_index, target))
                continue

            # raw() raises BufferError on non-contiguous buffers, matching
            # the rule enforced by the pickle machinery itself.
            with pickle_buffer.raw() as raw_view:
                readonly = raw_view.readonly
                length = raw_view.nbytes
                if length > max_frame_length:
                    raise BundleResourceLimit(
                        f"buffer at slot {slot_index} is {length} bytes long, "
                        f"exceeding max_frame_length={max_frame_length}"
                    )
                total_buffer_length += length
                if total_buffer_length + len(payload) > max_total_length:
                    raise BundleResourceLimit(
                        "total bundle size exceeds "
                        f"max_total_length={max_total_length}"
                    )
                digest = _digest_view(raw_view, chunk_size)
                flags = _FLAG_READONLY if readonly else 0
                stream.write(_U8.pack(_FRAME_BUFFER))
                stream.write(_U32.pack(slot_index))
                stream.write(_U8.pack(flags))
                stream.write(_U64.pack(length))
                stream.write(digest)
                for chunk in _iter_view_chunks(raw_view, chunk_size):
                    stream.write(chunk)
            first_slot_by_id[buffer_id] = slot_index
            manifest_entries.append(
                (
                    _MANIFEST_KIND_DATA,
                    slot_index,
                    flags,
                    length,
                    digest,
                )
            )

        # Completion manifest, published only once every frame succeeded.
        body = bytearray()
        body += _U32.pack(len(pickle_buffers))
        body += _U32.pack(len(first_slot_by_id))
        body += _U64.pack(len(payload))
        body += _U64.pack(total_buffer_length)
        body += payload_digest
        for entry in manifest_entries:
            if entry[0] == _MANIFEST_KIND_DATA:
                _kind, index, flags, length, digest = entry
                body += _U8.pack(_MANIFEST_KIND_DATA)
                body += _U32.pack(index)
                body += _U8.pack(flags)
                body += _U64.pack(length)
                body += digest
            else:
                _kind, index, target = entry
                body += _U8.pack(_MANIFEST_KIND_ALIAS)
                body += _U32.pack(index)
                body += _U32.pack(target)
        body_digest = hashlib.sha256(body).digest()
        stream.write(_U8.pack(_FRAME_MANIFEST))
        stream.write(_U64.pack(len(body)))
        stream.write(bytes(body))
        stream.write(body_digest)
        stream.write(_COMMIT_MAGIC)
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()
        committed = True
    finally:
        if owned:
            stream.close()
            if not committed:
                try:
                    os.unlink(file)
                except OSError:
                    pass
    # The caller-provided stream is intentionally never closed.


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------


def _read_header(stream):
    magic = _read_exact(stream, len(_MAGIC))
    if magic != _MAGIC:
        raise BundleFormatError(
            "bad bundle magic: the stream is not a cloudpickle bundle"
        )
    fixed = _read_exact(stream, 5)
    format_version, endian, pickle_protocol, digest_id, reserved_flags = (
        _U8.unpack(fixed[i : i + 1])[0] for i in range(5)
    )
    if format_version != BUNDLE_FORMAT_VERSION:
        raise BundleFormatError(
            f"unsupported bundle format version {format_version}; this "
            f"reader supports version {BUNDLE_FORMAT_VERSION}"
        )
    if endian != _ENDIAN_LE:
        # The byte order is explicit in the format: a different marker means
        # the stream was produced for an endianness this reader does not
        # support rather than something that may be silently reinterpreted.
        raise BundleFormatError(
            f"unsupported endianness marker {endian!r}; only little-endian "
            "bundles are supported"
        )
    if pickle_protocol < 5:
        raise BundleFormatError(
            f"bundle declares pickle protocol {pickle_protocol}, expected >= 5"
        )
    if digest_id != _DIGEST_SHA256:
        raise BundleFormatError(f"unsupported digest algorithm identifier {digest_id}")
    if reserved_flags != 0:
        raise BundleFormatError("non-zero reserved header flags")
    reserved = _read_exact(stream, 2)
    if _U16.unpack(reserved)[0] != 0:
        raise BundleFormatError("non-zero reserved header field")
    (producer_length,) = _U32.unpack(_read_exact(stream, 4))
    if producer_length > 4096:
        raise BundleFormatError(
            f"producer identity string is {producer_length} bytes long, "
            "exceeding 4096"
        )
    producer = _read_exact(stream, producer_length).decode("utf-8", errors="replace")
    declared_max_frame = _U64.unpack(_read_exact(stream, 8))[0]
    declared_max_total = _U64.unpack(_read_exact(stream, 8))[0]
    return {
        "format_version": format_version,
        "pickle_protocol": pickle_protocol,
        "producer": producer,
        "declared_max_frame_length": declared_max_frame,
        "declared_max_total_length": declared_max_total,
    }


def _stream_frame_data(stream, length, digest, sink, chunk_size, should_cancel):
    """Read exactly ``length`` bytes into ``sink`` while verifying ``digest``.

    The declared length is bounded by the caller before invoking this, so the
    loop performs bounded reads only. A short read, cancellation or digest
    mismatch raises and leaves the partially populated sink to the caller's
    cleanup.
    """
    hasher = hashlib.sha256()
    remaining = length
    while remaining:
        block = _read_exact(stream, min(remaining, chunk_size))
        if should_cancel is not None and should_cancel():
            raise BundleCancelled("bundle loading cancelled by caller")
        hasher.update(block)
        sink.write(block)
        remaining -= len(block)
    if not hmac.compare_digest(hasher.digest(), digest):
        raise BundleIntegrityError(
            "frame digest mismatch: the data does not match the digest "
            "recorded in its frame header"
        )


def _parse_manifest_body(body, context):
    """Parse and cross-check the committed manifest against the read frames."""
    offset = 0

    def take(size):
        nonlocal offset
        if offset + size > len(body):
            raise BundleFormatError("truncated manifest body")
        value = body[offset : offset + size]
        offset += size
        return value

    slot_count = _U32.unpack(take(4))[0]
    data_frame_count = _U32.unpack(take(4))[0]
    payload_length = _U64.unpack(take(8))[0]
    total_buffer_length = _U64.unpack(take(8))[0]
    payload_digest = take(_DIGEST_SIZE)

    entries = []
    seen_data = 0
    seen_alias = 0
    for _ in range(slot_count):
        kind = _U8.unpack(take(1))[0]
        if kind == _MANIFEST_KIND_DATA:
            index = _U32.unpack(take(4))[0]
            flags = _U8.unpack(take(1))[0]
            length = _U64.unpack(take(8))[0]
            digest = take(_DIGEST_SIZE)
            entries.append(("data", index, flags, length, digest))
            seen_data += 1
        elif kind == _MANIFEST_KIND_ALIAS:
            index = _U32.unpack(take(4))[0]
            target = _U32.unpack(take(4))[0]
            entries.append(("alias", index, target))
            seen_alias += 1
        else:
            raise BundleFormatError(f"unknown manifest entry kind {kind}")
    if offset != len(body):
        raise BundleFormatError("trailing bytes in manifest body")

    if seen_data != data_frame_count:
        raise BundleIntegrityError(
            f"manifest data frame count {data_frame_count} does not match "
            f"the {seen_data} data records it contains"
        )
    if slot_count != context["slot_count"]:
        raise BundleIntegrityError(
            f"manifest declares {slot_count} buffer slots but the stream "
            f"contains {context['slot_count']} buffer frames"
        )
    if payload_length != context["payload_length"]:
        raise BundleIntegrityError(
            "manifest payload length does not match the payload frame"
        )
    if payload_digest != context["payload_digest"]:
        raise BundleIntegrityError(
            "manifest payload digest does not match the payload frame digest"
        )
    if total_buffer_length != context["total_buffer_length"]:
        raise BundleIntegrityError(
            "manifest total buffer length does not match the buffer frames"
        )

    # Every manifest record must match the corresponding buffer frame, in
    # order. Payload frames are not part of the manifest entries.
    buffer_records = [
        record for record in context["frame_records"] if record[0] != "payload"
    ]
    if len(entries) != len(buffer_records):
        raise BundleIntegrityError(
            "manifest entry count does not match the number of buffer frames"
        )
    for position, (entry, record) in enumerate(zip(entries, buffer_records)):
        if entry[0] != record[0]:
            raise BundleIntegrityError(
                f"manifest entry {position} kind does not match frame"
            )
        if entry[0] == "data":
            _kind, index, flags, length, digest = entry
            _rkind, r_index, r_flags, r_length, r_digest = record
            if (index, flags, length, digest) != (
                r_index,
                r_flags,
                r_length,
                r_digest,
            ):
                raise BundleIntegrityError(
                    f"manifest record for buffer slot {index} does not match "
                    "its frame header"
                )
            if flags & ~_KNOWN_FLAGS:
                raise BundleFormatError(
                    f"buffer slot {index} uses unknown flag bits {flags}"
                )
        else:
            _kind, index, target = entry
            _rkind, r_index, r_target = record
            if (index, target) != (r_index, r_target):
                raise BundleIntegrityError(
                    f"manifest alias record at slot {index} does not match " "its frame"
                )
            if target >= index or target not in context["data_slot_indexes"]:
                raise BundleIntegrityError(
                    f"alias slot {index} points at invalid target {target}"
                )

    # The pickle payload itself is the authoritative source for the number
    # of out-of-band buffers it consumes. Counting NEXT_BUFFER opcodes runs
    # no reconstructor at all, yet catches missing or extra buffers before
    # unpickling starts.
    payload_sink = context["payload_sink"]
    payload_sink.seek(0)
    next_buffer_count = sum(
        1
        for opcode, _arg, _pos in pickletools.genops(payload_sink)
        if opcode.name == "NEXT_BUFFER"
    )
    if next_buffer_count != slot_count:
        raise BundleIntegrityError(
            f"pickle payload references {next_buffer_count} out-of-band "
            f"buffers but the bundle provides {slot_count}"
        )

    return {
        "slot_count": slot_count,
        "data_frame_count": data_frame_count,
        "alias_count": seen_alias,
        "payload_length": payload_length,
        "payload_digest": payload_digest,
        "total_buffer_length": total_buffer_length,
        "entries": entries,
    }


def load_bundle(
    file,
    *,
    chunk_size=DEFAULT_CHUNK_SIZE,
    spool_size=DEFAULT_SPOOL_SIZE,
    max_frame_length=DEFAULT_MAX_FRAME_LENGTH,
    max_total_length=DEFAULT_MAX_TOTAL_LENGTH,
    max_buffers=DEFAULT_MAX_BUFFERS,
    should_cancel=None,
):
    """Load and fully validate a bundle before reconstructing its object.

    All frame boundaries, declared lengths, digests, buffer counts and
    resource limits are verified while the stream is read in bounded chunks.
    Only after every check has succeeded is :func:`pickle.loads` (or an
    equivalent unpickler) invoked. A malformed bundle therefore never executes
    a reconstructor.

    Parameters
    ----------
    file : path-like or binary readable file-like object
        Source of the bundle. A file-like object is never closed; a path is
        opened and closed internally.
    chunk_size : int
        Maximum size of an individual read.
    spool_size : int
        Per-buffer threshold above which the buffer is spilled to a
        memory-mapped temporary file instead of being held in memory.
    max_frame_length, max_total_length, max_buffers : int
        Resource limits enforced *before* the corresponding data is read. A
        frame declaring an oversized length fails immediately without
        attempting to materialize it.
    should_cancel : callable, optional
        Optional zero-argument callable inspected between chunks and frames;
        a truthy return value aborts loading with :class:`BundleCancelled`
        and releases every temporary resource.

    Returns
    -------
    LoadedBundle
        A context manager owning the reconstructed object and every backing
        resource. Call :meth:`LoadedBundle.close` once the reconstructed
        object no longer references the bundle buffers.
    """
    _check_positive_int("chunk_size", chunk_size)
    _check_positive_int("spool_size", spool_size)
    _check_positive_int("max_frame_length", max_frame_length)
    _check_positive_int("max_total_length", max_total_length)
    _check_positive_int("max_buffers", max_buffers)

    stream, owned = _open_sink(file, "rb")
    # Every resource created while reading is registered here so that an
    # error at any point leaves nothing behind. The caller's stream itself is
    # never part of this list and therefore never closed.
    sinks = []
    try:
        header = _read_header(stream)

        payload_sink = None
        payload_length = None
        payload_digest = None
        # slot index -> (sink, readonly, length, digest) for data slots
        data_slots = {}
        frame_records = []
        total_data_length = 0
        next_slot = 0
        manifest_seen = False

        while True:
            if should_cancel is not None and should_cancel():
                raise BundleCancelled("bundle loading cancelled by caller")
            marker = stream.read(1)
            if not marker:
                if not manifest_seen:
                    raise BundleFormatError(
                        "stream ended before the completion manifest"
                    )
                break
            frame_type = _U8.unpack(marker)[0]

            if frame_type == _FRAME_MANIFEST:
                manifest_seen = True
                (body_length,) = _U64.unpack(_read_exact(stream, 8))
                # The manifest size is bounded by the declared buffer count.
                manifest_cap = 256 + max_buffers * (1 + 4 + 1 + 8 + _DIGEST_SIZE)
                if body_length > manifest_cap:
                    raise BundleResourceLimit(
                        f"manifest body declares {body_length} bytes, "
                        f"exceeding the limit derived from max_buffers "
                        f"({manifest_cap})"
                    )
                body = _read_exact(stream, body_length)
                if should_cancel is not None and should_cancel():
                    raise BundleCancelled("bundle loading cancelled by caller")
                body_digest = _read_exact(stream, _DIGEST_SIZE)
                commit = _read_exact(stream, len(_COMMIT_MAGIC))
                if commit != _COMMIT_MAGIC:
                    raise BundleFormatError(
                        "bundle is not committed: missing or bad commit marker"
                    )
                if not hmac.compare_digest(hashlib.sha256(body).digest(), body_digest):
                    raise BundleIntegrityError(
                        "manifest digest mismatch: the completion manifest "
                        "has been altered or truncated"
                    )
                trailing = stream.read(1)
                if trailing:
                    raise BundleFormatError("trailing bytes after the committed bundle")
                break

            if manifest_seen:
                raise BundleFormatError("frames found after the completion manifest")

            if frame_type == _FRAME_PAYLOAD:
                if payload_sink is not None:
                    raise BundleFormatError("duplicate payload frame")
                (length,) = _U64.unpack(_read_exact(stream, 8))
                digest = _read_exact(stream, _DIGEST_SIZE)
                if length > max_frame_length:
                    raise BundleResourceLimit(
                        f"payload frame declares {length} bytes, exceeding "
                        f"max_frame_length={max_frame_length}"
                    )
                if length > max_total_length:
                    raise BundleResourceLimit(
                        f"payload frame declares {length} bytes, exceeding "
                        f"max_total_length={max_total_length}"
                    )
                payload_sink = _SpooledSink(spool_size)
                sinks.append(payload_sink)
                _stream_frame_data(
                    stream,
                    length,
                    digest,
                    payload_sink,
                    chunk_size,
                    should_cancel,
                )
                payload_length = length
                payload_digest = digest
                total_data_length += length
                frame_records.append(("payload",))

            elif frame_type == _FRAME_BUFFER:
                if payload_sink is None:
                    raise BundleFormatError(
                        "buffer frame encountered before the payload frame"
                    )
                (index,) = _U32.unpack(_read_exact(stream, 4))
                flags = _U8.unpack(_read_exact(stream, 1))[0]
                (length,) = _U64.unpack(_read_exact(stream, 8))
                digest = _read_exact(stream, _DIGEST_SIZE)
                if index != next_slot:
                    raise BundleFormatError(
                        f"expected buffer slot {next_slot}, frame declares "
                        f"slot {index}"
                    )
                if flags & ~_KNOWN_FLAGS:
                    raise BundleFormatError(
                        f"buffer slot {index} uses unknown flag bits {flags}"
                    )
                if next_slot >= max_buffers:
                    raise BundleResourceLimit(
                        f"more than max_buffers={max_buffers} buffer slots"
                    )
                if length > max_frame_length:
                    raise BundleResourceLimit(
                        f"buffer slot {index} declares {length} bytes, "
                        f"exceeding max_frame_length={max_frame_length}"
                    )
                total_data_length += length
                if total_data_length > max_total_length:
                    raise BundleResourceLimit(
                        "declared frame lengths exceed "
                        f"max_total_length={max_total_length}"
                    )
                sink = _SpooledSink(spool_size)
                sinks.append(sink)
                _stream_frame_data(
                    stream,
                    length,
                    digest,
                    sink,
                    chunk_size,
                    should_cancel,
                )
                data_slots[index] = (
                    sink,
                    bool(flags & _FLAG_READONLY),
                    length,
                    digest,
                )
                frame_records.append(("data", index, flags, length, digest))
                next_slot += 1

            elif frame_type == _FRAME_ALIAS:
                if payload_sink is None:
                    raise BundleFormatError(
                        "alias frame encountered before the payload frame"
                    )
                if next_slot >= max_buffers:
                    raise BundleResourceLimit(
                        f"more than max_buffers={max_buffers} buffer slots"
                    )
                (index,) = _U32.unpack(_read_exact(stream, 4))
                (target,) = _U32.unpack(_read_exact(stream, 4))
                if index != next_slot:
                    raise BundleFormatError(
                        f"expected buffer slot {next_slot}, alias frame "
                        f"declares slot {index}"
                    )
                if target >= index or target not in data_slots:
                    raise BundleFormatError(
                        f"alias slot {index} points at invalid target " f"{target}"
                    )
                frame_records.append(("alias", index, target))
                next_slot += 1

            else:
                raise BundleFormatError(
                    f"unknown frame type marker {frame_type} at slot {next_slot}"
                )

        if not manifest_seen:
            raise BundleFormatError("bundle has no completion manifest")
        if payload_sink is None:
            raise BundleFormatError("bundle is missing the payload frame")

        context = {
            "slot_count": next_slot,
            "frame_records": frame_records,
            "payload_sink": payload_sink,
            "payload_length": payload_length,
            "payload_digest": payload_digest,
            "total_buffer_length": total_data_length - payload_length,
            "data_slot_indexes": set(data_slots),
        }
        manifest_info = _parse_manifest_body(body, context)

        # All validation succeeded: materialize stable backing storage and
        # build one PickleBuffer per distinct data slot. Alias slots repeat
        # the same object, mirroring pickle.loads(buffers=callback_list).
        resources = []
        try:
            buffers_by_slot = {}
            for index, (sink, readonly, _length, _digest) in sorted(data_slots.items()):
                backing = sink.materialize(readonly)
                pickle_buffer = pickle.PickleBuffer(backing)
                is_mmap = isinstance(backing, mmap.mmap)
                resources.append((pickle_buffer, backing, sink, is_mmap))
                buffers_by_slot[index] = pickle_buffer

            buffers = []
            for record in frame_records:
                if record[0] == "data":
                    buffers.append(buffers_by_slot[record[1]])
                elif record[0] == "alias":
                    buffers.append(buffers_by_slot[record[2]])

            manifest = BundleManifest(
                format_version=header["format_version"],
                pickle_protocol=header["pickle_protocol"],
                digest_name=_DIGEST_NAME,
                producer=header["producer"],
                slot_count=manifest_info["slot_count"],
                data_frame_count=manifest_info["data_frame_count"],
                alias_count=manifest_info["alias_count"],
                payload_length=manifest_info["payload_length"],
                payload_digest=manifest_info["payload_digest"].hex(),
                total_buffer_length=manifest_info["total_buffer_length"],
                entries=tuple(manifest_info["entries"]),
            )

            # Reconstruction only happens now that the whole bundle is
            # validated.
            if payload_sink.rolled:
                payload_sink.seek(0)
                obj = pickle.Unpickler(payload_sink, buffers=buffers).load()
            else:
                obj = pickle.loads(payload_sink.getvalue(), buffers=buffers)
        except BaseException:
            # Materialization or reconstruction failed partway: release the
            # PickleBuffers and mappings already created before letting the
            # outer handler close every sink.
            for pickle_buffer, backing, _sink, is_mmap in resources:
                try:
                    pickle_buffer.release()
                except ValueError:
                    pass
                if is_mmap:
                    try:
                        backing.close()
                    except (OSError, BufferError):
                        pass
            raise

        return LoadedBundle(obj, buffers, manifest, tuple(resources))
    except BaseException:
        # Validation or reconstruction failed: release every temporary
        # resource we created. The caller's stream is not touched.
        for sink in sinks:
            try:
                sink.close()
            except OSError:
                # Best-effort cleanup: the original failure must propagate.
                pass
        if owned:
            stream.close()
        raise
    else:
        if owned:
            stream.close()
