"""Tests for the versioned protocol 5 buffer bundle format.

The bundle layer is a pure container around ``cloudpickle`` protocol 5
pickling: every successful roundtrip is cross-checked against the equivalent
``pickle.loads(payload, buffers=callback_list)`` call, and every malformed
input must fail during validation, before any reconstructor runs.
"""

import hashlib
import io
import mmap
import pickle
from pickle import PickleBuffer

import pytest

import cloudpickle
from cloudpickle import (
    BUNDLE_FORMAT_VERSION,
    BundleCancelled,
    BundleError,
    BundleFormatError,
    BundleIntegrityError,
    BundleResourceLimit,
    dump_bundle,
    load_bundle,
)
from cloudpickle.bundle import (
    _COMMIT_MAGIC,
    _DIGEST_SHA256,
    _DIGEST_SIZE,
    _ENDIAN_LE,
    _FRAME_ALIAS,
    _FRAME_BUFFER,
    _FRAME_MANIFEST,
    _FRAME_PAYLOAD,
    _MAGIC,
    _MANIFEST_KIND_ALIAS,
    _MANIFEST_KIND_DATA,
    _U8,
    _U16,
    _U32,
    _U64,
)

# ----------------------------------------------------------------------------
# Module-level reconstructor whose executions can be counted. It must live at
# module scope so that the pickle stream can reference it by global name.
# ----------------------------------------------------------------------------

_RECONSTRUCTOR_CALLS = []


def counting_reconstructor(*buffers):
    """Reconstructor recording every invocation for zero-execution checks."""
    _RECONSTRUCTOR_CALLS.append(tuple(buffers))
    return CountingPayload(*buffers)


class CountingPayload:
    def __init__(self, *buffers):
        self.buffers = buffers


class CountingObject:
    """Object reducing to several out-of-band PickleBuffer arguments."""

    def __init__(self, *datas):
        self.datas = datas

    def __reduce__(self):
        return counting_reconstructor, tuple(PickleBuffer(d) for d in self.datas)


class RepeatedReferenceObject:
    """Same PickleBuffer object referenced several times by the payload."""

    def __reduce__(self):
        pb = PickleBuffer(b"alias-buffer-content-" * 64)
        return counting_reconstructor, (pb, pb, pb)


@pytest.fixture(autouse=True)
def _reset_reconstructor_calls():
    _RECONSTRUCTOR_CALLS.clear()
    yield
    _RECONSTRUCTOR_CALLS.clear()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _plain_loads_reference(obj):
    """Reference behavior: vanilla cloudpickle + pickle.loads(buffers=...)."""
    buffers = []
    payload = cloudpickle.dumps(obj, protocol=5, buffer_callback=buffers.append)
    return pickle.loads(payload, buffers=buffers), buffers


def _dump_bytes(obj, **kwargs):
    sink = io.BytesIO()
    dump_bundle(obj, sink, **kwargs)
    return sink.getvalue()


def _load(blob, **kwargs):
    return load_bundle(io.BytesIO(blob), **kwargs)


def _identity_groups(sequence):
    """Map each distinct object id to the positions where it appears."""
    groups = {}
    for position, item in enumerate(sequence):
        groups.setdefault(id(item), []).append(position)
    return sorted(groups.values())


class _WriteTrackingSink:
    """File-like sink recording the largest individual write() call."""

    def __init__(self):
        self._wrapped = io.BytesIO()
        self.max_write = 0
        self.write_sizes = []
        self.closed = False

    def write(self, data):
        size = len(data)
        self.max_write = max(self.max_write, size)
        self.write_sizes.append(size)
        return self._wrapped.write(data)

    def flush(self):
        self._wrapped.flush()

    def getvalue(self):
        return self._wrapped.getvalue()

    def close(self):
        self.closed = True


class _FailingAfterWrites:
    def __init__(self, fail_after):
        self._wrapped = io.BytesIO()
        self.remaining = fail_after
        self.published = b""

    def write(self, data):
        if self.remaining <= 0:
            raise OSError("simulated write failure")
        self.remaining -= 1
        return self._wrapped.write(data)

    def flush(self):
        pass

    @property
    def closed(self):
        return self._wrapped.closed


def _split_bundle_frames(blob):
    """White-box parser returning header bytes, frame blocks and manifest."""
    bio = io.BytesIO(blob)
    assert bio.read(len(_MAGIC)) == _MAGIC
    bio.read(5)
    bio.read(2)
    (producer_length,) = _U32.unpack(bio.read(4))
    bio.read(producer_length)
    bio.read(16)  # two u64 producer limits
    header_bytes = blob[: bio.tell()]
    blocks = []
    manifest_body = None
    while True:
        start = bio.tell()
        frame_type = _U8.unpack(bio.read(1))[0]
        if frame_type == _FRAME_PAYLOAD:
            (length,) = _U64.unpack(bio.read(8))
            digest = bio.read(_DIGEST_SIZE)
            data = bio.read(length)
            blocks.append(("payload", length, digest, data, blob[start : bio.tell()]))
        elif frame_type == _FRAME_BUFFER:
            (index,) = _U32.unpack(bio.read(4))
            flags = _U8.unpack(bio.read(1))[0]
            (length,) = _U64.unpack(bio.read(8))
            digest = bio.read(_DIGEST_SIZE)
            data = bio.read(length)
            blocks.append(
                (
                    "data",
                    index,
                    flags,
                    length,
                    digest,
                    data,
                    blob[start : bio.tell()],
                )
            )
        elif frame_type == _FRAME_ALIAS:
            (index,) = _U32.unpack(bio.read(4))
            (target,) = _U32.unpack(bio.read(4))
            blocks.append(("alias", index, target, blob[start : bio.tell()]))
        elif frame_type == _FRAME_MANIFEST:
            (body_length,) = _U64.unpack(bio.read(8))
            manifest_body = bio.read(body_length)
            bio.read(_DIGEST_SIZE)
            assert bio.read(len(_COMMIT_MAGIC)) == _COMMIT_MAGIC
            break
        else:  # pragma: no cover - defensive, malformed inputs tested elsewhere
            raise AssertionError(f"unexpected frame type {frame_type}")
    return header_bytes, blocks, manifest_body


def _repack(header_bytes, blocks, manifest_body):
    """Re-emit a bundle with a replaced manifest and a fresh body digest."""
    out = bytearray(header_bytes)
    for block in blocks:
        out += block[-1]
    out += _U8.pack(_FRAME_MANIFEST)
    out += _U64.pack(len(manifest_body))
    out += manifest_body
    out += hashlib.sha256(manifest_body).digest()
    out += _COMMIT_MAGIC
    return bytes(out)


def _parse_manifest_entries(body):
    # Two u32 counts and two u64 lengths precede the payload digest.
    offset = 24 + _DIGEST_SIZE
    entries = []
    while offset < len(body):
        kind = _U8.unpack(body[offset : offset + 1])[0]
        offset += 1
        if kind == _MANIFEST_KIND_DATA:
            index = _U32.unpack(body[offset : offset + 4])[0]
            offset += 4
            flags = _U8.unpack(body[offset : offset + 1])[0]
            offset += 1
            length = _U64.unpack(body[offset : offset + 8])[0]
            offset += 8
            digest = body[offset : offset + _DIGEST_SIZE]
            offset += _DIGEST_SIZE
            entries.append(("data", index, flags, length, digest))
        else:
            assert kind == _MANIFEST_KIND_ALIAS
            index = _U32.unpack(body[offset : offset + 4])[0]
            offset += 4
            target = _U32.unpack(body[offset : offset + 4])[0]
            offset += 4
            entries.append(("alias", index, target))
    return entries


def _build_manifest_body(
    slot_count, data_count, payload_length, total_length, payload_digest, entries
):
    body = bytearray()
    body += _U32.pack(slot_count)
    body += _U32.pack(data_count)
    body += _U64.pack(payload_length)
    body += _U64.pack(total_length)
    body += payload_digest
    for entry in entries:
        if entry[0] == "data":
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
    return bytes(body)


def manifest_total(manifest_body):
    # Layout: u32 slot_count, u32 data_count, u64 payload_length (offset 8),
    # u64 total_buffer_length (offset 16).
    return _U64.unpack(manifest_body[16:24])[0]


# ----------------------------------------------------------------------------
# Happy paths and equivalence with plain pickle.loads(buffers=...)
# ----------------------------------------------------------------------------


def test_custom_proto5_object_roundtrip_matches_plain_loads():
    obj = CountingObject(
        bytearray(b"writable-buffer-one"),
        b"readonly-buffer-two",
        bytearray(b"writable-buffer-three"),
    )
    blob = _dump_bytes(obj, protocol=5)
    bundle = _load(blob)
    try:
        plain, plain_buffers = _plain_loads_reference(obj)
        assert len(bundle.buffers) == len(plain_buffers) == 3
        assert _RECONSTRUCTOR_CALLS, "reconstructor must run on valid bundle"
        for got, expected in zip(bundle.object.buffers, plain.buffers):
            assert bytes(got) == bytes(expected)
        # Read-only / writable semantics survive the container.
        with memoryview(bundle.buffers[0]) as view:
            assert view.readonly is False
        with memoryview(bundle.buffers[1]) as view:
            assert view.readonly is True
        with memoryview(bundle.buffers[2]) as view:
            assert view.readonly is False
    finally:
        bundle.object = None
        bundle.close()


def test_repeated_references_become_aliases_preserving_identity():
    obj = RepeatedReferenceObject()
    blob = _dump_bytes(obj, protocol=5)
    bundle = _load(blob)
    try:
        manifest = bundle.manifest
        assert manifest.slot_count == 3
        assert manifest.data_frame_count == 1
        assert manifest.alias_count == 2
        # The three slots expose the very same PickleBuffer object.
        assert bundle.buffers[0] is bundle.buffers[1] is bundle.buffers[2]

        plain, plain_buffers = _plain_loads_reference(obj)
        assert len(plain_buffers) == 3
        assert plain_buffers[0] is plain_buffers[1] is plain_buffers[2]
        assert _identity_groups(bundle.buffers) == _identity_groups(plain_buffers)
        assert bytes(bundle.object.buffers[0]) == bytes(plain.buffers[0])
    finally:
        bundle.object = None
        bundle.close()


def test_distinct_buffers_sharing_memory_remain_distinct():
    raw = bytearray(b"shared-backing-memory")
    # Two separate PickleBuffer objects wrap the same writable memory.
    obj = CountingObject(raw, raw)
    blob = _dump_bytes(obj, protocol=5)
    bundle = _load(blob)
    try:
        # Distinct PickleBuffer objects stay distinct, as in plain pickle.
        _plain, plain_buffers = _plain_loads_reference(obj)
        assert bundle.buffers[0] is not bundle.buffers[1]
        assert plain_buffers[0] is not plain_buffers[1]
        assert bytes(bundle.buffers[0]) == bytes(bundle.buffers[1]) == raw
    finally:
        bundle.object = None
        bundle.close()


def test_empty_buffers_are_supported():
    obj = CountingObject(bytearray(), b"", bytearray(b"x"))
    blob = _dump_bytes(obj, protocol=5)
    bundle = _load(blob)
    try:
        with memoryview(bundle.buffers[0]) as view:
            assert view.nbytes == 0 and view.readonly is False
        with memoryview(bundle.buffers[1]) as view:
            assert view.nbytes == 0 and view.readonly is True
        with memoryview(bundle.buffers[2]) as view:
            assert view.nbytes == 1
        assert bundle.manifest.total_buffer_length == 1
    finally:
        bundle.object = None
        bundle.close()


def test_bundle_without_any_buffer():
    obj = {"no": ["out", "of", "band"], "n": 42}
    blob = _dump_bytes(obj)
    bundle = _load(blob)
    try:
        assert bundle.manifest.slot_count == 0
        assert bundle.buffers == ()
        assert bundle.object == obj
    finally:
        bundle.close()


def test_header_carries_protocol_and_implementation_identity():
    blob = _dump_bytes(CountingObject(b"abc"), protocol=5)
    bundle = _load(blob)
    try:
        manifest = bundle.manifest
        assert manifest.format_version == BUNDLE_FORMAT_VERSION
        assert manifest.pickle_protocol == 5
        assert manifest.digest_name == "sha256"
        assert "cloudpickle" in manifest.producer
        assert "CPython" in manifest.producer or "PyPy" in manifest.producer
        assert (
            manifest.payload_digest
            == hashlib.sha256(_split_bundle_frames(blob)[1][0][3]).hexdigest()
        )
    finally:
        bundle.object = None
        bundle.close()


def test_protocol_below_five_is_rejected():
    with pytest.raises(ValueError, match="protocol >= 5"):
        dump_bundle(CountingObject(b"x"), io.BytesIO(), protocol=4)


def test_writable_buffer_remains_writable_after_roundtrip():
    payload = bytearray(b"mutable")
    obj = CountingObject(payload)
    blob = _dump_bytes(obj, protocol=5)
    bundle = _load(blob)
    try:
        with bundle.buffers[0].raw() as view:
            assert view.readonly is False
            view[0] = ord("M")
        with bundle.buffers[0].raw() as view:
            assert bytes(view) == b"Mutable"
    finally:
        bundle.object = None
        bundle.close()


# ----------------------------------------------------------------------------
# Real PickleBuffer support rules
# ----------------------------------------------------------------------------


def test_non_contiguous_buffer_raises_pickling_error():
    strided = memoryview(bytearray(range(40)))[::2]
    assert strided.contiguous is False
    obj = CountingObject(strided)
    with pytest.raises(pickle.PicklingError, match="non-contiguous"):
        dump_bundle(obj, io.BytesIO(), protocol=5)
    assert _RECONSTRUCTOR_CALLS == []


def test_numpy_roundtrip_and_writability(tmp_path):
    np = pytest.importorskip("numpy")
    array = np.arange(1_000_000, dtype=np.int64).reshape(1000, 1000)
    obj = {"array": array, "meta": ("answer", 42)}

    path = tmp_path / "numpy_bundle.bin"
    with open(path, "wb") as sink:
        dump_bundle(obj, sink, chunk_size=128 * 1024)

    bundle = load_bundle(path, chunk_size=128 * 1024, spool_size=256 * 1024)
    try:
        loaded_array = bundle.object["array"]
        assert np.array_equal(loaded_array, array)
        assert loaded_array.flags["WRITEABLE"] == array.flags["WRITEABLE"]
        assert bundle.object["meta"] == ("answer", 42)

        plain, _ = _plain_loads_reference(obj)
        assert np.array_equal(plain["array"], loaded_array)
    finally:
        # Drop every view over the mmap storage before closing the bundle.
        del loaded_array
        bundle.object = None
        bundle.close()


# ----------------------------------------------------------------------------
# Corruption and fail-fast validation (zero reconstructor execution)
# ----------------------------------------------------------------------------


def test_corrupted_last_frame_fails_integrity_check():
    obj = CountingObject(b"first-buffer", b"CORRUPT-ME-MARKER" * 32)
    blob = _dump_bytes(obj, protocol=5)
    marker_position = blob.rfind(b"CORRUPT-ME-MARKER")
    tampered = bytearray(blob)
    tampered[marker_position] ^= 0xFF
    with pytest.raises(BundleIntegrityError, match="digest"):
        _load(bytes(tampered))
    assert _RECONSTRUCTOR_CALLS == []


def test_manifest_digest_tampering_is_rejected():
    blob = _dump_bytes(CountingObject(b"abc"), protocol=5)
    tampered = bytearray(blob)
    # The manifest body digest starts right before the 9-byte commit marker.
    digest_start = len(tampered) - len(_COMMIT_MAGIC) - _DIGEST_SIZE
    tampered[digest_start] ^= 0x01
    with pytest.raises(BundleIntegrityError, match="manifest digest"):
        _load(bytes(tampered))
    assert _RECONSTRUCTOR_CALLS == []


def test_bad_commit_marker_means_unpublished_manifest():
    blob = _dump_bytes(CountingObject(b"abc"), protocol=5)
    tampered = bytearray(blob)
    tampered[-1] ^= 0x01
    with pytest.raises(BundleFormatError, match="commit"):
        _load(bytes(tampered))
    assert _RECONSTRUCTOR_CALLS == []


def test_truncated_stream_is_short_read_error():
    obj = CountingObject(b"abc", b"def" * 1000)
    blob = _dump_bytes(obj, protocol=5)
    with pytest.raises(BundleFormatError, match="end of stream"):
        _load(blob[: len(blob) - 40])

    # Cutting exactly at the manifest frame marker leaves no manifest at all.
    header_bytes, blocks, _manifest = _split_bundle_frames(blob)
    manifest_offset = len(header_bytes) + sum(len(block[-1]) for block in blocks)
    with pytest.raises(BundleFormatError, match="completion manifest"):
        _load(blob[:manifest_offset])
    assert _RECONSTRUCTOR_CALLS == []


def test_non_bundle_input_and_trailing_bytes():
    with pytest.raises(BundleFormatError, match="magic"):
        _load(b"this is definitely not a bundle")
    blob = _dump_bytes(CountingObject(b"abc"), protocol=5)
    with pytest.raises(BundleFormatError, match="trailing"):
        _load(blob + b"\x00extra")
    assert _RECONSTRUCTOR_CALLS == []


def test_unsupported_version_and_endian_markers_are_rejected():
    blob = bytearray(_dump_bytes(CountingObject(b"abc"), protocol=5))
    bad_version = bytearray(blob)
    bad_version[8] = 99
    with pytest.raises(BundleFormatError, match="format version"):
        _load(bytes(bad_version))

    bad_endian = bytearray(blob)
    bad_endian[9] = ord(">")
    with pytest.raises(BundleFormatError, match="endianness"):
        _load(bytes(bad_endian))
    assert _RECONSTRUCTOR_CALLS == []


class _SmallReadStream:
    """Readable wrapper that never returns more than ``max_block`` bytes."""

    def __init__(self, data, max_block):
        self._buffer = io.BytesIO(data)
        self._max_block = max_block

    def read(self, size=-1):
        if size is None or size < 0:
            size = self._max_block
        return self._buffer.read(min(size, self._max_block))

    def readline(self):
        return self._buffer.readline(self._max_block)


def test_load_supports_streams_returning_small_reads():
    obj = CountingObject(b"abc" * 500, b"def" * 500)
    blob = _dump_bytes(obj, chunk_size=128)
    bundle = load_bundle(
        _SmallReadStream(blob, max_block=37), chunk_size=128, spool_size=256
    )
    try:
        assert [bytes(b) for b in bundle.buffers] == [b"abc" * 500, b"def" * 500]
    finally:
        bundle.object = None
        bundle.close()


def test_huge_declared_length_is_rejected_before_reading():
    huge = 10**18
    header = b"".join(
        (
            _MAGIC,
            _U8.pack(BUNDLE_FORMAT_VERSION),
            _U8.pack(_ENDIAN_LE),
            _U8.pack(5),
            _U8.pack(_DIGEST_SHA256),
            _U8.pack(0),
            _U16.pack(0),
            _U32.pack(0),
            _U64.pack(huge),
            _U64.pack(huge),
        )
    )
    # A payload frame claiming an absurd length, followed by nothing.
    malicious = header + _U8.pack(_FRAME_PAYLOAD) + _U64.pack(huge) + b"\x00" * 32
    with pytest.raises(BundleResourceLimit, match="max_frame_length"):
        _load(malicious, max_frame_length=1 << 30)
    assert _RECONSTRUCTOR_CALLS == []

    # A length that fits per-frame limits but breaks the aggregate budget.
    over_total = (1 << 30) + 1
    malicious_total = (
        header + _U8.pack(_FRAME_PAYLOAD) + _U64.pack(over_total) + b"\x00" * 32
    )
    with pytest.raises(BundleResourceLimit, match="max_total_length"):
        _load(
            malicious_total,
            max_frame_length=1 << 32,
            max_total_length=1 << 30,
        )
    assert _RECONSTRUCTOR_CALLS == []


def test_declared_length_with_truncated_data_does_not_allocate():
    obj = CountingObject(b"abc")
    blob = _dump_bytes(obj, protocol=5)
    header_bytes, blocks, _manifest = _split_bundle_frames(blob)
    payload_block = blocks[0]
    _kind, length, digest, _data, _raw = payload_block
    # Same header digest, but declare far more data than the stream contains.
    forged_payload = (
        _U8.pack(_FRAME_PAYLOAD) + _U64.pack(length + 4096) + digest + _data
    )
    forged = header_bytes + forged_payload
    with pytest.raises(BundleFormatError, match="end of stream"):
        _load(forged, max_frame_length=1 << 30)
    assert _RECONSTRUCTOR_CALLS == []


def _swap_payload(blob_with_buffers, blob_with_payload):
    """Replace the payload frame while keeping buffer frames and records."""
    header_bytes, blocks, manifest_body = _split_bundle_frames(blob_with_buffers)
    _other_header, other_blocks, _other_manifest = _split_bundle_frames(
        blob_with_payload
    )
    new_payload_block = other_blocks[0]
    entries = _parse_manifest_entries(manifest_body)
    data_entries = [e for e in entries if e[0] == "data"]
    new_manifest = _build_manifest_body(
        slot_count=len(entries),
        data_count=len(data_entries),
        payload_length=new_payload_block[1],
        total_length=manifest_total(manifest_body),
        payload_digest=new_payload_block[2],
        entries=entries,
    )
    swapped_blocks = [new_payload_block] + blocks[1:]
    return _repack(header_bytes, swapped_blocks, new_manifest)


def test_payload_expecting_fewer_buffers_is_rejected():
    two_buffers = _dump_bytes(CountingObject(b"buffer-zero", b"buffer-one"), protocol=5)
    one_buffer = _dump_bytes(CountingObject(b"only-one-buffer"), protocol=5)
    # Frames and manifest consistently provide 2 slots, but the swapped
    # payload only consumes 1 NEXT_BUFFER: an extra buffer is detected.
    rebuilt = _swap_payload(two_buffers, one_buffer)
    with pytest.raises(BundleIntegrityError, match="out-of-band"):
        _load(rebuilt)
    assert _RECONSTRUCTOR_CALLS == []


def test_payload_expecting_more_buffers_is_rejected():
    one_buffer = _dump_bytes(CountingObject(b"only-one-buffer"), protocol=5)
    two_buffers = _dump_bytes(CountingObject(b"buffer-zero", b"buffer-one"), protocol=5)
    # The swapped payload consumes 2 NEXT_BUFFER while the bundle provides 1:
    # a missing buffer is detected before unpickling.
    rebuilt = _swap_payload(one_buffer, two_buffers)
    with pytest.raises(BundleIntegrityError, match="out-of-band"):
        _load(rebuilt)
    assert _RECONSTRUCTOR_CALLS == []


def test_max_buffers_limit_enforced_before_manifest():
    obj = CountingObject(b"a", b"b", b"c")
    blob = _dump_bytes(obj, protocol=5)
    with pytest.raises(BundleResourceLimit, match="max_buffers"):
        _load(blob, max_buffers=2)
    assert _RECONSTRUCTOR_CALLS == []

    # Alias slots count towards the limit as well.
    alias_blob = _dump_bytes(RepeatedReferenceObject(), protocol=5)
    with pytest.raises(BundleResourceLimit, match="max_buffers"):
        _load(alias_blob, max_buffers=2)
    assert _RECONSTRUCTOR_CALLS == []


# ----------------------------------------------------------------------------
# Streaming, resource ownership and cleanup semantics
# ----------------------------------------------------------------------------


def test_chunked_file_sink_and_mmap_spill_roundtrip(tmp_path):
    obj = CountingObject(
        bytearray(b"x" * (300 * 1024)),
        b"y" * (200 * 1024),
        b"small",
    )
    path = tmp_path / "chunked.bundle"
    with open(path, "wb") as sink:
        dump_bundle(obj, sink, chunk_size=4096)

    bundle = load_bundle(path, chunk_size=4096, spool_size=64 * 1024)
    try:
        # The two large buffers must have spilled to mmap-backed storage.
        mmap_resources = [r for r in bundle._resources if r[3]]
        assert len(mmap_resources) == 2
        assert all(isinstance(r[1], mmap.mmap) for r in mmap_resources)
        for got, expected in zip(bundle.object.buffers, obj.datas):
            assert bytes(got) == bytes(expected)
    finally:
        bundle.object = None
        bundle.close()
    assert bundle.closed


def test_writable_mmap_backing_is_actually_mutable(tmp_path):
    blob = _dump_bytes(
        CountingObject(bytearray(b"mutable-mmap-buffer" * 20000)),
        chunk_size=4096,
    )
    path = tmp_path / "writable.bundle"
    path.write_bytes(blob)
    bundle = load_bundle(path, chunk_size=4096, spool_size=64 * 1024)
    try:
        assert any(resource[3] for resource in bundle._resources)
        with bundle.buffers[0].raw() as view:
            assert view.readonly is False
            view[0] = ord("M")
        with bundle.buffers[0].raw() as view:
            assert view[0] == ord("M")
    finally:
        bundle.object = None
        bundle.close()


def test_aliases_share_spilled_mmap_storage(tmp_path):
    blob = _dump_bytes(RepeatedReferenceObject(), chunk_size=4096)
    path = tmp_path / "alias.bundle"
    path.write_bytes(blob)
    bundle = load_bundle(path, chunk_size=4096, spool_size=1024)
    try:
        assert bundle.manifest.alias_count == 2
        assert bundle.buffers[0] is bundle.buffers[1] is bundle.buffers[2]
        resource = bundle._resources[0]
        assert resource[3] is True  # the single backing storage is mmap'd
        assert bundle.manifest.data_frame_count == 1
    finally:
        bundle.object = None
        bundle.close()


def test_caller_stream_is_never_closed():
    sink = io.BytesIO()
    dump_bundle(CountingObject(b"abc"), sink)
    assert not sink.closed

    source = io.BytesIO(sink.getvalue())
    bundle = load_bundle(source)
    bundle.object = None
    bundle.close()
    assert not source.closed

    bad_source = io.BytesIO(b"not a bundle")
    with pytest.raises(BundleFormatError):
        load_bundle(bad_source)
    assert not bad_source.closed


def test_failed_write_does_not_publish_manifest_and_keeps_stream_open():
    failing = _FailingAfterWrites(fail_after=2)
    with pytest.raises(OSError, match="simulated write failure"):
        dump_bundle(
            CountingObject(b"a" * 1024, b"b" * 1024),
            failing,
            chunk_size=256,
        )
    assert not failing.closed
    written = failing._wrapped.getvalue()
    assert _COMMIT_MAGIC not in written
    assert _U8.pack(_FRAME_MANIFEST) not in written


def test_owned_path_is_removed_after_failed_dump(tmp_path):
    target = tmp_path / "incomplete.bundle"
    # A resource-limit breach happens while frames are being written to the
    # internally opened destination, which must then be closed and removed.
    with pytest.raises(BundleResourceLimit):
        dump_bundle(
            CountingObject(b"x" * (4 * 1024 * 1024)),
            target,
            chunk_size=64 * 1024,
            max_total_length=1024,
        )
    assert not target.exists()


def test_cancellation_cleans_temp_resources(tmp_path):
    obj = CountingObject(*(b"buffer-%04d" % i * 4096 for i in range(8)))
    path = tmp_path / "cancel.bundle"
    with open(path, "wb") as sink:
        dump_bundle(obj, sink, chunk_size=4096)

    state = {"checks": 0}

    def should_cancel():
        state["checks"] += 1
        return state["checks"] > 4

    with pytest.raises(BundleCancelled):
        load_bundle(
            path,
            chunk_size=4096,
            spool_size=64 * 1024,
            should_cancel=should_cancel,
        )
    assert _RECONSTRUCTOR_CALLS == []
    # The input stream (owned path) is closed but the file remains available;
    # no loader-created temporary file survives the cancelled call.
    assert path.exists()


def test_close_with_live_view_raises_instead_of_dangling(tmp_path):
    blob = _dump_bytes(CountingObject(bytearray(b"z" * (256 * 1024))), chunk_size=4096)
    path = tmp_path / "owned.bundle"
    path.write_bytes(blob)
    bundle = load_bundle(path, chunk_size=4096, spool_size=64 * 1024)
    view = memoryview(bundle.buffers[0])
    with pytest.raises(BundleError, match="still exports views"):
        bundle.close()
    assert bundle.closed is False
    view.release()
    bundle.object = None
    bundle.close()
    assert bundle.closed is True
    # Closing twice is a no-op.
    bundle.close()


def test_context_manager_closes_bundle():
    blob = _dump_bytes(CountingObject(b"abc"))
    with load_bundle(io.BytesIO(blob)) as bundle:
        assert bytes(bundle.object.buffers[0]) == b"abc"
        bundle.object = None
    assert bundle.closed


# ----------------------------------------------------------------------------
# Peak memory behavior
# ----------------------------------------------------------------------------


def test_dump_does_not_concatenate_buffers_into_one_write():
    chunk_size = 64 * 1024
    datas = [bytearray(b"%d" % i) * (512 * 1024) for i in range(4)]
    aggregate = sum(map(len, datas))
    tracker = _WriteTrackingSink()
    dump_bundle(CountingObject(*datas), tracker, chunk_size=chunk_size)
    assert tracker.max_write <= chunk_size
    assert all(size < aggregate for size in tracker.write_sizes)
    # Sanity check: the produced bundle still loads correctly.
    bundle = _load(tracker.getvalue())
    try:
        for got, expected in zip(bundle.object.buffers, datas):
            assert bytes(got) == bytes(expected)
    finally:
        bundle.object = None
        bundle.close()


def test_load_peak_extra_memory_stays_bounded():
    tracemalloc = pytest.importorskip("tracemalloc")
    buffer_size = 1024 * 1024
    datas = [bytearray(b"%d" % i) * buffer_size for i in range(4)]
    aggregate = len(datas) * buffer_size
    blob = _dump_bytes(CountingObject(*datas), chunk_size=64 * 1024)
    source = io.BytesIO(blob)

    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        bundle = load_bundle(source, chunk_size=64 * 1024, spool_size=128 * 1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    try:
        # Every buffer exceeded the spool threshold and lives in an mmap,
        # whose pages are not Python heap allocations: traced peak extra
        # memory must stay well below the aggregate buffer size.
        assert all(resource[3] for resource in bundle._resources)
        assert peak - baseline < aggregate // 4
        for got, expected in zip(bundle.object.buffers, datas):
            assert bytes(got) == bytes(expected)
    finally:
        bundle.object = None
        bundle.close()


def test_in_memory_path_avoids_mmap_and_still_validates():
    obj = CountingObject(b"small-a", b"small-b")
    blob = _dump_bytes(obj, chunk_size=1024)
    bundle = _load(blob, spool_size=1 << 20)
    try:
        assert not any(resource[3] for resource in bundle._resources)
        assert [bytes(b) for b in bundle.buffers] == [b"small-a", b"small-b"]
    finally:
        bundle.object = None
        bundle.close()


def test_payload_spilled_to_disk_uses_file_backed_unpickler():
    obj = CountingObject(b"abc", b"def")
    blob = _dump_bytes(obj, chunk_size=256)
    # A spool threshold smaller than the payload forces the payload itself
    # onto a temporary file, exercising the file-backed unpickler branch.
    bundle = load_bundle(io.BytesIO(blob), chunk_size=256, spool_size=8)
    try:
        assert [bytes(b) for b in bundle.buffers] == [b"abc", b"def"]
        assert bundle.object.buffers[1] is bundle.buffers[1]
    finally:
        bundle.object = None
        bundle.close()


def _minimal_header():
    return b"".join(
        (
            _MAGIC,
            _U8.pack(BUNDLE_FORMAT_VERSION),
            _U8.pack(_ENDIAN_LE),
            _U8.pack(5),
            _U8.pack(_DIGEST_SHA256),
            _U8.pack(0),
            _U16.pack(0),
            _U32.pack(0),
            _U64.pack(1 << 30),
            _U64.pack(1 << 30),
        )
    )


def test_reconstruction_failure_after_validation_cleans_resources():
    # A validly framed bundle whose payload references an unresolvable
    # global: every container check passes, then unpickling itself fails.
    bad_payload = b"\x80\x05cos.path\ndoes_not_exist_attr\n."
    digest = hashlib.sha256(bad_payload).digest()
    raw_block = (
        _U8.pack(_FRAME_PAYLOAD) + _U64.pack(len(bad_payload)) + digest + bad_payload
    )
    payload_block = ("payload", len(bad_payload), digest, bad_payload, raw_block)
    manifest_body = _build_manifest_body(
        slot_count=0,
        data_count=0,
        payload_length=len(bad_payload),
        total_length=0,
        payload_digest=digest,
        entries=[],
    )
    blob = _repack(_minimal_header(), [payload_block], manifest_body)
    with pytest.raises(AttributeError):
        _load(blob, spool_size=8)
    # The failure happens during unpickling, so no reconstructor ever runs.
    assert _RECONSTRUCTOR_CALLS == []
