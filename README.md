# cloudpickle

[![Automated Tests](https://github.com/cloudpipe/cloudpickle/workflows/Automated%20Tests/badge.svg?branch=master&event=push)](https://github.com/cloudpipe/cloudpickle/actions)
[![codecov.io](https://codecov.io/github/cloudpipe/cloudpickle/coverage.svg?branch=master)](https://codecov.io/github/cloudpipe/cloudpickle?branch=master)

`cloudpickle` makes it possible to serialize Python constructs not supported
by the default `pickle` module from the Python standard library.

`cloudpickle` is especially useful for **cluster computing** where Python
code is shipped over the network to execute on remote hosts, possibly close
to the data.

Among other things, `cloudpickle` supports pickling for **lambda functions**
along with **functions and classes defined interactively** in the
`__main__` module (for instance in a script, a shell or a Jupyter notebook).

Cloudpickle can only be used to send objects between the **exact same version
of Python**.

Using `cloudpickle` for **long-term object storage is not supported and
strongly discouraged.**

**Security notice**: one should **only load pickle data from trusted sources** as
otherwise `pickle.load` can lead to arbitrary code execution resulting in a critical
security vulnerability.


Installation
------------

The latest release of `cloudpickle` is available from
[pypi](https://pypi.python.org/pypi/cloudpickle):

    pip install cloudpickle


Examples
--------

Pickling a lambda expression:

```python
>>> import cloudpickle
>>> squared = lambda x: x ** 2
>>> pickled_lambda = cloudpickle.dumps(squared)

>>> import pickle
>>> new_squared = pickle.loads(pickled_lambda)
>>> new_squared(2)
4
```

Pickling a function interactively defined in a Python shell session
(in the `__main__` module):

```python
>>> CONSTANT = 42
>>> def my_function(data: int) -> int:
...     return data + CONSTANT
...
>>> pickled_function = cloudpickle.dumps(my_function)
>>> depickled_function = pickle.loads(pickled_function)
>>> depickled_function
<function __main__.my_function(data:int) -> int>
>>> depickled_function(43)
85
```


Overriding pickle's serialization mechanism for importable constructs:
----------------------------------------------------------------------

An important difference between `cloudpickle` and `pickle` is that
`cloudpickle` can serialize a function or class **by value**, whereas `pickle`
can only serialize it **by reference**. Serialization by reference treats
functions and classes as attributes of modules, and pickles them through
instructions that trigger the import of their module at load time.
Serialization by reference is thus limited in that it assumes that the module
containing the function or class is available/importable in the unpickling
environment. This assumption breaks when pickling constructs defined in an
interactive session, a case that is automatically detected by `cloudpickle`,
that pickles such constructs **by value**.

Another case where the importability assumption is expected to break is when
developing a module in a distributed execution environment: the worker
processes may not have access to the said module, for example if they live on a
different machine than the process in which the module is being developed. By
itself, `cloudpickle` cannot detect such "locally importable" modules and
switch to serialization by value; instead, it relies on its default mode, which
is serialization by reference. However, since `cloudpickle 2.0.0`, one can
explicitly specify modules for which serialization by value should be used,
using the
`register_pickle_by_value(module)`/`/unregister_pickle_by_value(module)` API:

```python
>>> import cloudpickle
>>> import my_module
>>> cloudpickle.register_pickle_by_value(my_module)
>>> cloudpickle.dumps(my_module.my_function)  # my_function is pickled by value
>>> cloudpickle.unregister_pickle_by_value(my_module)
>>> cloudpickle.dumps(my_module.my_function)  # my_function is pickled by reference
```

Using this API, there is no need to re-install the new version of the module on
all the worker nodes nor to restart the workers: restarting the client Python
process with the new source code is enough.

Note that this feature is still **experimental**, and may fail in the following
situations:

- If the body of a function/class pickled by value contains an `import` statement:
  ```python
  >>> def f():
  >>> ... from another_module import g
  >>> ... # calling f in the unpickling environment may fail if another_module
  >>> ... # is unavailable
  >>> ... return g() + 1
  ```

- If a function pickled by reference uses a function pickled by value during its execution.


Externalizing global bindings with resource tokens
---------------------------------------------------

Some globals read by by-value functions cannot (or should not) be serialized
by value: live database connections, sockets, client handles, ... A
`GlobalBindingPolicy` makes it possible to replace such a binding in the
pickle stream with an opaque, restricted **pure-data token** (only `None`,
`bool`, `int`, `float`, `str`, `bytes` and nested `tuple`/`list`/`set`/
`frozenset`/`dict` of those are allowed). The bound value itself never enters
the payload.

Selection is based on the identity of the function's globals namespace
(`func.__globals__`) and on the binding name, never on the identity of the
bound value, so the same object stored in an unrelated container (or in
another globals namespace) is still serialized by value:

```python
>>> import cloudpickle

>>> namespace = {"db_connection": live_connection}  # live_connection is unpicklable
>>> exec("def query():\n    return db_connection", namespace)
>>> policy = cloudpickle.StaticGlobalBindingPolicy(
...     [(namespace, "db_connection", "resource-id-1234")]
... )
>>> payload = cloudpickle.dumps(namespace["query"], global_binding_policy=policy)
```

On the loading side, an explicit resolver callable turns each token back into
an object. It is called as `resolver(token, binding)`, where `binding` is a
`GlobalBinding` describing the location of the binding; resolver errors are
reported with that binding path. Tokens are never interpreted by cloudpickle
itself (no implicit import, network access or file read):

```python
>>> def resolver(token, binding):
...     assert binding.name == "db_connection"
...     return reconnect_using_resource_id(token)

>>> loaded_query = cloudpickle.loads(
...     payload, global_binding_resolver=resolver
... )
```

The sharing rule is fixed by the protocol: each `(globals namespace, binding
name)` pair is resolved exactly once per load, and every function sharing that
namespace receives the same resolved object. Functions sharing a namespace
must externalize a given name with a consistent token; inconsistent decisions
raise a `pickle.PicklingError` before any byte is written. The original
`func.__globals__` is never modified, functions pickled by reference are not
affected by the policy, and loading only writes into the freshly reconstructed
by-value globals dict — real imported module `__dict__` objects are never
polluted. Streams containing externalized bindings cannot be loaded with a
plain `pickle.loads`: an explicit error message points to
`cloudpickle.loads(..., global_binding_resolver=...)`.


Running the tests
-----------------

- With `tox`, to test run the tests for all the supported versions of
  Python and PyPy:

      pip install tox
      tox

  or alternatively for a specific environment:

      tox -e py312


- With `pytest` to only run the tests for your current version of
  Python:

      pip install -r dev-requirements.txt
      PYTHONPATH='.:tests' pytest

History
-------

`cloudpickle` was initially developed by [picloud.com](http://web.archive.org/web/20140721022102/http://blog.picloud.com/2013/11/17/picloud-has-joined-dropbox/) and shipped as part of
the client SDK.

A copy of `cloudpickle.py` was included as part of PySpark, the Python
interface to [Apache Spark](https://spark.apache.org/). Davies Liu, Josh
Rosen, Thom Neale and other Apache Spark developers improved it significantly,
most notably to add support for PyPy and Python 3.

The aim of the `cloudpickle` project is to make that work available to a wider
audience outside of the Spark ecosystem and to make it easier to improve it
further notably with the help of a dedicated non-regression test suite.
