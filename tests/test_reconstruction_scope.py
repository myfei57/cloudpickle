"""Tests for the scoped reconstruction of dynamic objects.

``cloudpickle.ReconstructionScope`` provides a local identity namespace for
the dynamic classes, Enum subclasses and TypeVars that cloudpickle pickles by
value. These tests check that:

- payloads loaded through the same scope share reconstructed dynamic objects
  while two distinct scopes (and the default global loading mode) stay
  mutually distinct;
- the process-global dynamic class trackers are never read from nor written
  to by scoped loads, so concurrent plain loads and existing object identities
  are preserved;
- scope activation is local to threads and asyncio tasks and nested explicit
  activations are properly restored;
- the scope only keeps weak references, rejects new loads once closed, leaves
  already returned objects usable and does not pollute ``sys.modules``;
- old pickle streams and payloads without tracker ids keep loading.
"""

import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import gc
import io
import logging
import pickle
from pathlib import Path
import sys
import threading
import typing
import weakref

import pytest

import cloudpickle
from cloudpickle import ReconstructionScope
from cloudpickle.cloudpickle import (
    _ACTIVE_RECONSTRUCTION_SCOPE,
    _DYNAMIC_CLASS_TRACKER_BY_CLASS,
    _DYNAMIC_CLASS_TRACKER_BY_ID,
    _lookup_class_or_track,
)


def _make_dynamic_payloads():
    """Build independent payloads sharing dynamic definitions.

    All the classes below are defined inside this function so that cloudpickle
    treats them as dynamic (pickled by value), while the two subclass payloads
    still carry the same tracker id for their shared base.
    """

    class Base:
        label = "base"

        def greet(self):
            return f"hello-{self.label}"

    class SubA(Base):
        label = "a"

        def add(self, x):
            return x + 1

    class SubB(Base):
        label = "b"

        def make_base(self):
            # ``Base`` is resolved through the reconstructed function globals
            # and must point at the scope-local Base class.
            return Base()

    class Node:
        """Recursive dynamic class whose method closes over the class itself."""

        def __init__(self, value, child=None):
            self.value = value
            self.child = child

        def make_child(self, value):
            # ``Node`` is a cell variable of this function: the closure cell
            # must be filled with the scope-consistent Node class at load.
            return Node(value, self)

    return {
        "suba": cloudpickle.dumps(SubA()),
        "subb": cloudpickle.dumps(SubB()),
        "base": cloudpickle.dumps(Base),
        "nodes": cloudpickle.dumps((Node(1), Node(2))),
        "node_cls": cloudpickle.dumps(Node),
    }


def _make_enum_payload():
    class Color(Enum):
        RED = 1
        BLUE = 2

    return cloudpickle.dumps((Color, Color.RED, Color.BLUE))


def _make_typevar_payload():
    T = typing.TypeVar("T", bound=int)

    class Box(typing.Generic[T]):
        def wrap(self, item):
            return item

    return cloudpickle.dumps((T, Box)), T, Box


def test_same_payload_two_scopes_and_default_loads_have_distinct_identities():
    payloads = _make_dynamic_payloads()
    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    a1 = scope_a.loads(payloads["suba"])
    a2 = scope_a.loads(payloads["suba"])
    b1 = scope_b.loads(payloads["suba"])
    default_1 = cloudpickle.loads(payloads["suba"])
    default_2 = cloudpickle.loads(payloads["suba"])

    cls_a = type(a1)
    # Repeated loads in the same scope reuse one class object.
    assert type(a2) is cls_a
    # The two independent scopes reconstruct mutually distinct new classes.
    assert type(b1) is not cls_a
    # The default loading mode keeps its existing process-global behavior and
    # is itself independent from any scope.
    cls_global = type(default_1)
    assert type(default_2) is cls_global
    assert cls_global is not cls_a
    assert cls_global is not type(b1)

    # isinstance and actual method behavior follow the scope-local identity.
    assert isinstance(a1, cls_a)
    assert isinstance(a2, cls_a)
    assert not isinstance(a1, type(b1))
    assert not isinstance(b1, cls_a)
    assert a1.greet() == "hello-a"
    assert a1.add(41) == 42
    assert b1.greet() == "hello-a"


def test_same_scope_shared_base_across_independent_payloads():
    payloads = _make_dynamic_payloads()
    scope = ReconstructionScope()

    sub_a = scope.loads(payloads["suba"])
    sub_b = scope.loads(payloads["subb"])
    base_direct = scope.loads(payloads["base"])

    base_a = type(sub_a).__mro__[1]
    base_b = type(sub_b).__mro__[1]
    # The base class embedded in two independent payloads is reconstructed
    # once and shared inside the scope.
    assert base_a is base_b is base_direct
    assert isinstance(sub_a, base_direct)
    assert isinstance(sub_b, base_direct)

    # The function globals of a reconstructed method resolve to the same
    # scope-local Base definition.
    assert type(sub_b.make_base()) is base_direct

    # A second scope does not share any of these class objects.
    other_scope = ReconstructionScope()
    other_sub_a = other_scope.loads(payloads["suba"])
    assert type(other_sub_a) is not type(sub_a)
    assert type(other_sub_a).__mro__[1] is not base_direct


def test_single_payload_multi_instance_recursive_class_and_closure_identity():
    payloads = _make_dynamic_payloads()
    scope = ReconstructionScope()

    node1, node2 = scope.loads(payloads["nodes"])
    node_cls = scope.loads(payloads["node_cls"])

    # Multiple instances of the same dynamic class in one payload, plus a
    # class object coming from another payload, all map to one scope class.
    assert type(node1) is type(node2) is node_cls
    assert isinstance(node1, node_cls) and isinstance(node2, node_cls)

    # The recursive constructor and the closure cell point at that same class.
    child = node1.make_child(3)
    assert type(child) is node_cls
    assert child.child is node1
    assert child.value == 3

    # Everything stays consistent when the same payload is loaded again.
    node1_bis, node2_bis = scope.loads(payloads["nodes"])
    assert type(node1_bis) is node_cls
    assert type(node1_bis.make_child(4)) is node_cls


def test_enum_tracking_is_scoped():
    payload = _make_enum_payload()
    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    enum_a1, red_a1, blue_a1 = scope_a.loads(payload)
    enum_a2, red_a2, blue_a2 = scope_a.loads(payload)
    enum_b, red_b, blue_b = scope_b.loads(payload)
    enum_default = cloudpickle.loads(payload)[0]

    assert enum_a1 is enum_a2
    assert enum_a1 is not enum_b
    assert enum_a1 is not enum_default
    # Enum members follow the scope-local enum class identity.
    assert red_a1 is red_a2 is enum_a1.RED
    assert blue_a1 is blue_a2 is enum_a1.BLUE
    assert red_a1 is not red_b
    assert red_b is enum_b.RED
    assert red_a1.value == 1 and blue_a1.name == "BLUE"
    assert isinstance(red_a1, enum_a1)
    assert not isinstance(red_a1, enum_b)


def test_typevar_and_generic_tracking_are_scoped():
    payload, original_t, original_box = _make_typevar_payload()
    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    t_a1, box_a1 = scope_a.loads(payload)
    t_a2, box_a2 = scope_a.loads(payload)
    t_b, box_b = scope_b.loads(payload)
    t_default, box_default = cloudpickle.loads(payload)

    assert t_a1 is t_a2
    assert t_a1 is not t_b
    assert t_a1 is not t_default
    assert t_a1.__name__ == original_t.__name__
    assert t_a1.__bound__ is int
    assert box_a1 is box_a2
    assert box_a1 is not box_b and box_a1 is not box_default
    # The reconstructed generic class keeps working with the scope TypeVar.
    parameterized = box_a1[t_a1]
    assert parameterized.__origin__ is box_a1
    assert box_a1().wrap("ok") == "ok"


def test_scoped_loads_never_touch_the_process_global_trackers():
    payloads = _make_dynamic_payloads()
    enum_payload = _make_enum_payload()
    tv_payload, _, _ = _make_typevar_payload()

    # Populate the global trackers first with a default load.
    global_obj = cloudpickle.loads(payloads["suba"])
    global_cls = type(global_obj)
    global_ids_before = dict(_DYNAMIC_CLASS_TRACKER_BY_ID)
    global_classes_before = dict(_DYNAMIC_CLASS_TRACKER_BY_CLASS)
    assert global_cls in _DYNAMIC_CLASS_TRACKER_BY_CLASS

    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()
    scoped_a = scope_a.loads(payloads["suba"])
    scope_a.loads(enum_payload)
    scope_a.loads(tv_payload)
    scope_b.loads(payloads["suba"])
    scope_b.loads(enum_payload)

    # The global mappings are untouched by the isolated loads: same keys and
    # the exact same class objects as before the scoped loads.
    assert dict(_DYNAMIC_CLASS_TRACKER_BY_ID) == global_ids_before
    assert dict(_DYNAMIC_CLASS_TRACKER_BY_CLASS) == global_classes_before

    # None of the reconstructed scope objects are registered globally.
    scoped_classes = [
        type(scoped_a),
        type(scope_b.loads(payloads["suba"])),
        scope_a.loads(enum_payload)[0],
        scope_b.loads(enum_payload)[0],
        scope_a.loads(tv_payload)[0],
    ]
    for obj in scoped_classes:
        assert obj not in _DYNAMIC_CLASS_TRACKER_BY_CLASS
        assert obj not in _DYNAMIC_CLASS_TRACKER_BY_ID.values()

    # Concurrent plain loads still resolve to the stable global identity.
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(lambda _: type(cloudpickle.loads(payloads["suba"])), range(20))
        )
    assert all(cls is global_cls for cls in results)


def test_by_reference_objects_are_not_copied_by_scopes():
    # Importable classes are pickled by reference and must bypass the scope
    # trackers entirely, staying shared process-wide.
    payload = cloudpickle.dumps((logging.Logger, collections.OrderedDict))
    scope = ReconstructionScope()
    logger_cls, ordered_dict_cls = scope.loads(payload)
    assert logger_cls is logging.Logger
    assert ordered_dict_cls is collections.OrderedDict
    assert logging.Logger not in scope._tracker_by_class

    test_pkg = pytest.importorskip("_cloudpickle_testpkg")
    pkg_payload = cloudpickle.dumps(test_pkg.package_function)
    assert scope.loads(pkg_payload) is test_pkg.package_function


def test_close_rejects_new_loads_but_keeps_returned_objects_usable():
    payloads = _make_dynamic_payloads()
    enum_payload = _make_enum_payload()
    scope = ReconstructionScope()
    assert not scope.closed

    obj = scope.loads(payloads["suba"])
    sub_b = scope.loads(payloads["subb"])
    enum_cls, red, _ = scope.loads(enum_payload)
    obj.greet()  # usable before closing

    scope.close()
    assert scope.closed
    # Closing twice is harmless.
    scope.close()

    with pytest.raises(RuntimeError, match="closed ReconstructionScope"):
        scope.loads(payloads["suba"])
    with pytest.raises(RuntimeError, match="closed ReconstructionScope"):
        scope.load(io.BytesIO(payloads["suba"]))
    with pytest.raises(RuntimeError, match="closed ReconstructionScope"):
        with scope:
            pass

    # Objects and classes returned before close keep working and keep their
    # identity; close does not invalidate anything.
    assert obj.greet() == "hello-a"
    assert obj.add(1) == 2
    assert red is enum_cls.RED
    assert isinstance(obj, type(obj))
    assert type(sub_b.make_base()).__name__ == "Base"


def test_scope_uses_weak_ownership_without_unbounded_strong_references():
    payloads = _make_dynamic_payloads()
    scope = ReconstructionScope()
    obj = scope.loads(payloads["suba"])
    cls = type(obj)
    cls_ref = weakref.ref(cls)

    assert len(scope._tracker_by_id) >= 1
    del obj, cls
    gc.collect()

    # The scope does not own strong references: the reconstructed class is
    # collected and the weak tracker entries are reclaimed.
    assert cls_ref() is None
    assert len(scope._tracker_by_id) == 0
    assert len(scope._tracker_by_class) == 0

    # A later load then builds a fresh class instead of resurrecting a dead one.
    new_obj = scope.loads(payloads["suba"])
    assert cls_ref() is None
    assert new_obj.greet() == "hello-a"


def test_concurrent_reconstruction_in_threads():
    payloads = _make_dynamic_payloads()
    enum_payload = _make_enum_payload()
    tv_payload, _, _ = _make_typevar_payload()

    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    # Preload one canonical object per scope so worker threads only have to
    # check identity stability.
    cls_a = type(scope_a.loads(payloads["suba"]))
    cls_b = type(scope_b.loads(payloads["suba"]))
    enum_a = scope_a.loads(enum_payload)[0]
    tv_a = scope_a.loads(tv_payload)[0]
    cls_global = type(cloudpickle.loads(payloads["suba"]))

    errors = []

    def worker(index):
        try:
            scope = scope_a if index % 2 == 0 else scope_b
            expected_cls = cls_a if scope is scope_a else cls_b
            for _ in range(10):
                obj = scope.loads(payloads["suba"])
                assert type(obj) is expected_cls
                assert isinstance(obj, expected_cls)
                assert obj.greet() == "hello-a"
                if scope is scope_a:
                    assert scope.loads(enum_payload)[0] is enum_a
                    assert scope.loads(tv_payload)[0] is tv_a
                # Plain loads inside worker threads stay global.
                assert type(cloudpickle.loads(payloads["suba"])) is cls_global
        except BaseException as exc:  # pragma: no cover - reported back
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(32)))

    assert errors == []
    # The two scopes remain isolated from each other and from the global mode.
    assert cls_a is not cls_b
    assert cls_a is not cls_global and cls_b is not cls_global
    scope_a.close()
    scope_b.close()


def test_scope_activation_does_not_leak_between_threads():
    scope = ReconstructionScope()
    entered = threading.Event()
    release = threading.Event()
    seen_in_thread = []

    def target():
        with scope:
            seen_in_thread.append(_ACTIVE_RECONSTRUCTION_SCOPE.get())
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=target)
    thread.start()
    try:
        assert entered.wait(5)
        # The scope activated in the worker thread must not be visible here.
        assert _ACTIVE_RECONSTRUCTION_SCOPE.get() is None
    finally:
        release.set()
        thread.join()
    assert seen_in_thread == [scope]
    assert _ACTIVE_RECONSTRUCTION_SCOPE.get() is None


def test_scope_activation_does_not_leak_between_asyncio_tasks():
    payloads = _make_dynamic_payloads()
    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    async def task(scope, recorded):
        with scope:
            recorded.append(_ACTIVE_RECONSTRUCTION_SCOPE.get())
            await asyncio.sleep(0)
            obj = scope.loads(payloads["suba"])  # nested explicit activation
            await asyncio.sleep(0)
            recorded.append(_ACTIVE_RECONSTRUCTION_SCOPE.get())
        assert _ACTIVE_RECONSTRUCTION_SCOPE.get() is None
        return type(obj)

    async def main():
        recorded_a, recorded_b = [], []
        cls_a, cls_b = await asyncio.gather(
            task(scope_a, recorded_a), task(scope_b, recorded_b)
        )
        assert recorded_a == [scope_a, scope_a]
        assert recorded_b == [scope_b, scope_b]
        assert cls_a is not cls_b
        assert _ACTIVE_RECONSTRUCTION_SCOPE.get() is None

    asyncio.run(main())


def test_nested_explicit_activation_is_restored():
    payloads = _make_dynamic_payloads()
    outer = ReconstructionScope()
    inner = ReconstructionScope()
    get_active = _ACTIVE_RECONSTRUCTION_SCOPE.get

    assert get_active() is None
    with outer:
        assert get_active() is outer
        # Plain cloudpickle.loads inside an activated block routes to it.
        routed = cloudpickle.loads(payloads["suba"])
        direct = outer.loads(payloads["suba"])
        assert type(routed) is type(direct)

        # Nested explicit override via another scope's loads.
        other = inner.loads(payloads["suba"])
        assert type(other) is not type(direct)
        assert get_active() is outer  # restored after the nested call

        with inner:
            assert get_active() is inner
        assert get_active() is outer

        # A failing nested load still restores the outer scope.
        with pytest.raises(Exception):
            inner.loads(b"not a pickle stream")
        assert get_active() is outer

    assert get_active() is None


def test_scope_load_reads_from_file_like_objects():
    payloads = _make_dynamic_payloads()
    scope = ReconstructionScope()
    stream = io.BytesIO(payloads["nodes"])
    node1, node2 = scope.load(stream)
    assert type(node1) is type(node2)
    assert type(node1) is type(scope.loads(payloads["nodes"])[0])


def test_scope_does_not_pollute_sys_modules_or_class_names():
    payloads = _make_dynamic_payloads()
    scope = ReconstructionScope()
    modules_before = set(sys.modules)

    obj = scope.loads(payloads["suba"])
    cls = type(obj)

    assert set(sys.modules) == modules_before
    assert scope._scope_id not in cls.__name__
    assert scope._scope_id not in cls.__qualname__
    assert repr(scope) == f"<ReconstructionScope {scope._scope_id} (open)>"


def test_payloads_without_tracker_ids_are_not_tracked():
    sentinel = object()
    scope = ReconstructionScope()
    # A None tracker id (legacy streams predating dynamic class tracking)
    # must bypass every tracker, including scope-local ones.
    assert _lookup_class_or_track(None, sentinel) is sentinel
    assert len(scope._tracker_by_id) == 0
    assert len(scope._tracker_by_class) == 0

    # The same must hold when such a stream is reconstructed in a scope:
    # activating the scope does not turn None ids into tracked entries.
    token = _ACTIVE_RECONSTRUCTION_SCOPE.set(scope)
    try:
        assert _lookup_class_or_track(None, sentinel) is sentinel
    finally:
        _ACTIVE_RECONSTRUCTION_SCOPE.reset(token)
    assert len(scope._tracker_by_id) == 0


def test_old_pickle_streams_load_inside_scopes():
    # Historical streams are only guaranteed loadable on the same Python
    # version that generated them, mirroring tests/test_backward_compat.py.
    python_info = "{}_{}{}".format(
        sys.implementation.name, sys.version_info.major, sys.version_info.minor
    )
    simple_class_pkl = (
        Path(__file__).parent / "old_pickles" / python_info / "simple_class.pkl"
    )
    if not simple_class_pkl.exists():
        pytest.skip(f"No historical pickle stream for {python_info}")

    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()
    global_ids_before = dict(_DYNAMIC_CLASS_TRACKER_BY_ID)

    with open(simple_class_pkl, "rb") as f:
        cls_a = scope_a.load(f)
    instance = cls_a(1)
    assert instance.attribute == 1

    # Unknown historical tracker ids are adopted by the scope and reused on
    # every later occurrence, including in another load of the same stream.
    with open(simple_class_pkl, "rb") as f:
        cls_a_bis = scope_a.load(f)
    assert cls_a_bis is cls_a

    with open(simple_class_pkl, "rb") as f:
        cls_b = scope_b.load(f)
    assert cls_b is not cls_a

    # Scoped loading of historical streams leaves the global trackers alone,
    # while the original plain pickle.load helper still works as before.
    assert dict(_DYNAMIC_CLASS_TRACKER_BY_ID) == global_ids_before
    with open(simple_class_pkl, "rb") as f:
        cls_global = pickle.load(f)
    assert cls_global().attribute == 1
    assert cls_global is not cls_a and cls_global is not cls_b


def test_scope_works_with_protocol2_payloads():
    class Local:
        def ping(self):
            return "pong"

    payload = cloudpickle.dumps(Local(), protocol=2)
    scope_a = ReconstructionScope()
    scope_b = ReconstructionScope()

    first = scope_a.loads(payload)
    assert type(first) is type(scope_a.loads(payload))
    assert type(first) is not type(scope_b.loads(payload))
    assert first.ping() == "pong"
    assert isinstance(first, type(first))


def test_scope_loads_accepts_out_of_band_buffers():
    class OOBContainer:
        def __init__(self, raw):
            self.raw = raw

        def __reduce_ex__(self, protocol):
            if protocol >= 5:
                return (OOBContainer, (pickle.PickleBuffer(self.raw),))
            return super().__reduce_ex__(protocol)

    def dump_with_buffers():
        buffers = []
        payload = cloudpickle.dumps(
            OOBContainer(bytearray(b"out-of-band payload")),
            protocol=5,
            buffer_callback=buffers.append,
        )
        assert len(buffers) == 1
        return payload, buffers

    scope = ReconstructionScope()
    payload, buffers = dump_with_buffers()
    first = scope.loads(payload, buffers=buffers)
    # A fresh dump carries the same tracker id and must resolve to the same
    # scope-local class through a separate set of out-of-band buffers.
    payload, buffers = dump_with_buffers()
    second = scope.loads(payload, buffers=buffers)

    assert type(first) is type(second)
    assert bytes(first.raw) == b"out-of-band payload"
    assert bytes(second.raw) == b"out-of-band payload"
    # The out-of-band reconstruction did not register anything globally.
    assert type(first) not in _DYNAMIC_CLASS_TRACKER_BY_CLASS
