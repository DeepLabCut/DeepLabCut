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

"""Base classes for benchmark and result definition

Benchmarks subclass the abstract ``Benchmark`` class and are defined by ``name``, their
``keypoints`` names, as well as groundtruth and metadata necessary to run evaluation.
Right now, the metrics to compute and report for each of the multi-animal benchmarks is the
root mean-squared-error (RMSE) and the mean average precision (mAP).

Note for contributors: If you decide to contribute a benchmark which does not fit
into this evaluation framework, please feel free to extend the base classes
(e.g. to support additional metrics).
"""

import abc
import dataclasses
from typing import Iterable
from typing import Tuple

import pandas as pd

import deeplabcut.benchmark.metrics
from deeplabcut import __version__


class BenchmarkEvaluationError(RuntimeError):
    pass


class Benchmark(abc.ABC):
    """Abstract benchmark baseclass.

    All benchmarks should subclass this class.
    """

    @abc.abstractmethod
    def names(self):
        """A unique key to describe this submission, e.g. the model name.

        This is also the name that will later appear in the benchmark table.
        The name needs to be unique across the whole benchmark. Non-unique names
        will raise an error during submission of a PR.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_predictions(self):
        """Return predictions for all images in the benchmark."""
        raise NotImplementedError()

    def __init__(self):
        keys = ["name", "keypoints", "ground_truth", "metadata"]
        for key in keys:
            if not hasattr(self, key):
                raise NotImplementedError(
                    f"Subclass of abstract Benchmark class need "
                    f"to define the {key} property."
                )

    def compute_pose_rmse(self, results_objects):
        return deeplabcut.benchmark.metrics.calc_rmse_from_obj(
            results_objects, h5_file=self.ground_truth, metadata_file=self.metadata
        )

    def compute_pose_map(self, results_objects):
        return deeplabcut.benchmark.metrics.calc_map_from_obj(
            results_objects, h5_file=self.ground_truth, metadata_file=self.metadata
        )

    def evaluate(self, name: str, on_error="raise"):
        """Evaluate this benchmark with all registered methods."""

        if name not in self.names():
            raise ValueError(
                f"{name} is not registered. Valid names are {self.names()}"
            )
        if on_error not in ("ignore", "return", "raise"):
            raise ValueError(f"on_error got an undefined value: {on_error}")
        mean_avg_precision = float("nan")
        root_mean_squared_error = float("nan")
        try:
            predictions = self.get_predictions(name)
            mean_avg_precision = self.compute_pose_map(predictions)
            root_mean_squared_error = self.compute_pose_rmse(predictions)
        except Exception as exception:
            if on_error == "ignore":
                # ignore the exception and continue with the next evaluation, without
                # yielding a result value.
                return
            elif on_error == "return":
                # return the result value, with NaN as the result for all metrics that
                # could not be computed due to the error.
                pass
            elif on_error == "raise":
                # raise the error and stop evaluation
                raise BenchmarkEvaluationError(
                    f"Error during benchmark evaluation for model {name}"
                ) from exception
            else:
                raise NotImplementedError() from exception
        return Result(
            method_name=name,
            benchmark_name=self.name,
            mean_avg_precision=mean_avg_precision,
            root_mean_squared_error=root_mean_squared_error,
        )


@dataclasses.dataclass
class Result:
    """Benchmark result."""

    method_name: str
    benchmark_name: str
    root_mean_squared_error: float = float("nan")
    mean_avg_precision: float = float("nan")
    benchmark_version: str = __version__

    _export_mapping = dict(
        benchmark_name="benchmark",
        method_name="method",
        benchmark_version="version",
        root_mean_squared_error="RMSE",
        mean_avg_precision="mAP",
    )

    _primary_key = ("benchmark_name", "method_name", "benchmark_version")

    @property
    def primary_key(self) -> Tuple[str]:
        """The primary key to uniquely identify this result."""
        return tuple(getattr(self, k) for k in self._primary_key)

    @property
    def primary_key_names(self) -> Tuple[str]:
        """Names of the primary keys"""
        return tuple(self._export_mapping.get(k) for k in self._primary_key)

    def __str__(self):
        return (
            f"{self.method_name}, {self.benchmark_name}: "
            f"{self.mean_avg_precision} mAP, "
            f"{self.root_mean_squared_error} RMSE"
        )

    @classmethod
    def fromdict(cls, data: dict):
        """Construct result object from dictionary."""
        kwargs = {attr: data[key] for attr, key in cls._export_mapping.items()}
        return cls(**kwargs)

    def todict(self) -> dict:
        """Export result object to dictionary, with less verbose key names."""
        return {key: getattr(self, attr) for attr, key in self._export_mapping.items()}


class ResultCollection:
    def __init__(self, *results):
        self.results = {result.primary_key: result for result in results}

    @property
    def primary_key_names(self):
        return next(iter(self.results.values())).primary_key_names

    def toframe(self) -> pd.DataFrame:
        """Convert results to pandas dataframe"""
        return pd.DataFrame(
            [result.todict() for result in self.results.values()]
        ).set_index(list(self.primary_key_names))

    def add(self, result: Result):
        """Add a result to the collection."""
        if result.primary_key in self.results:
            raise ValueError(
                "An entry for {result.primary_key} does already "
                "exist in this collection. Did you try to add the "
                "same result twice?"
            )
        if len(self) > 0:
            if result.primary_key_names != self.primary_key_names:
                raise ValueError("Incompatible result format.")
        self.results[result.primary_key] = result

    @classmethod
    def fromdicts(cls, data: Iterable[dict]):
        return cls(*[Result.fromdict(entry) for entry in data])

    def todicts(self):
        return [result.todict() for result in self.results.values()]

    def __len__(self):
        return len(self.results)

    def __contains__(self, other: Result):
        if not isinstance(other, Result):
            raise ValueError(
                f"{type(self)} can only store objects of type Result, "
                f"but got {type(other)}."
            )
        return other.primary_key in self.results

    def __eq__(self, other):
        if not isinstance(other, ResultCollection):
            return False
        return other.results == self.results
