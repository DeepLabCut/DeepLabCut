#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#


import json
import os
from typing import Container
from typing import Literal

from deeplabcut.benchmark.base import Benchmark, Result, ResultCollection

DATA_ROOT = os.path.join(os.getcwd(), "data")
CACHE = os.path.join(os.getcwd(), ".results")

__registry = []


def register(cls):
    """Add a benchmark to the list of evaluations to run.

    Apply this function as a decorator to a class. Note that the
    class needs to be a subclass of the ``benchmark.base.Benchmark``
    base class.

    In most situations, it will be a subclass of one of the pre-defined
    benchmarks in ``benchmark.benchmarks``.

    Throws:
        ``ValueError`` if the decorator is applied to a class that is
        not a subclass of ``benchmark.base.Benchmark``.
    """
    if not issubclass(cls, Benchmark):
        raise ValueError(
            f"Can only register subclasses of {type(Benchmark)}, " f"but got {cls}."
        )
    __registry.append(cls)


def evaluate(
    include_benchmarks: Container[str] = None,
    results: ResultCollection = None,
    on_error="return",
) -> ResultCollection:
    """Run evaluation for all benchmarks and methods.

    Note that in order for your custom benchmark to be included during
    evaluation, the following conditions need to be met:

        - The benchmark subclassed one of the benchmark definitions in
          in ``benchmark.benchmarks``
        - The benchmark is registered by applying the ``@benchmark.register``
          decorator to the class
        - The benchmark was imported. This is done automatically for all
          benchmarks that are defined in submodules or subpackages of the
          ``benchmark.submissions`` module. For all other locations, make
          sure to manually import the packages **before** calling the
          ``evaluate()`` function.

    Args:
        include_benchmarks:
            If ``None``, run all benchmarks that were discovered. If a container
            is passed, only include methods that were defined on benchmarks with
            the specified names. E.g., ``include_benchmarks = ["trimouse"]`` would
            only evaluate methods of the trimouse benchmark dataset.
        on_error:
            see documentation in ``benchmark.base.Benchmark.evaluate()``

    Returns:
        A collection of all results, which can be printed or exported to
        ``pd.DataFrame`` or ``json`` file formats.
    """
    if results is None:
        results = ResultCollection()
    for benchmark_cls in __registry:
        if include_benchmarks is not None:
            if benchmark_cls.name not in include_benchmarks:
                continue
        benchmark = benchmark_cls()
        for name in benchmark.names():
            if Result(method_name=name, benchmark_name=benchmark_cls.name) in results:
                continue
            else:
                result = benchmark.evaluate(name, on_error=on_error)
                results.add(result)
    return results


def get_filepath(basename: str):
    return os.path.join(DATA_ROOT, basename)


def savecache(results: ResultCollection):
    with open(CACHE, "w") as fh:
        json.dump(results.todicts(), fh, indent=2)


def loadcache(
    cache=CACHE, on_missing: Literal["raise", "ignore"] = "ignore"
) -> ResultCollection:
    if not os.path.exists(cache):
        if on_missing == "raise":
            raise FileNotFoundError(cache)
        return ResultCollection()
    with open(cache, "r") as fh:
        try:
            data = json.load(fh)
        except json.decoder.JSONDecodeError as e:
            if on_missing == "raise":
                raise e
            return ResultCollection()
    return ResultCollection.fromdicts(data)
