# DeepLabCut benchmark

For further information and the leaderboard, see [the official homepage](https://benchmark.deeplabcut.org/).

## High Level API

When implementing your own benchmarks, the most important functions are directly accessible
under the ``deeplabcut.benchmark`` package.

```{eval-rst}
.. automodule:: deeplabcut.benchmark
   :members:
   :show-inheritance:
```

## Available benchmark definitions

See [the official benchmark page](https://benchmark.deeplabcut.org/datasets.html) for a full overview
of the available datasets. A benchmark submission should contain a result for at least one of these
benchmarks. For an example of how to implement a benchmark submission, refer to the baselines in the
[DeepLabCut benchmark repo](https://github.com/DeepLabCut/benchmark/tree/main/benchmark/baselines).

```{eval-rst}
.. automodule:: deeplabcut.benchmark.benchmarks
   :members:
   :show-inheritance:
```

## Metric calculation

```{eval-rst}
.. automodule:: deeplabcut.benchmark.metrics
   :members:
   :show-inheritance:
```
