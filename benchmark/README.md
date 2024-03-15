# Benchmarking with DeepLabCut

This folder contains a few scripts that can be very useful when benchmarking datasets
with DeepLabCut. But first, some definitions:

**Shuffle:** As always in DeepLabCut, a shuffle is an experiment. It has an index, and

**Split:** A split (or data split) is a partition of labeled images into a train
and test set. Each shuffle has a split.

## Creating Data Splits

Data splits can be created with the `create_train_test_splits.py` file. This script can
create an arbitrary number of train/test splits for a project (or group of projects), 
which can be very useful for 
[k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).
The main method has the following signature:

```python
def main(
    projects: list[Project],
    seeds: list[int],
    num_splits: int,
    train_fractions: list[float],
    output_file: Path,
) -> None:
    """Creates train/test splits for DeepLabCut projects

    Args:
        projects: projects for which to create train/test splits
        seeds: random seed to use for each project (must be the same len as projects)
        num_splits: the number of train/test splits to create for each project
        train_fractions: the train fractions for which to create train/test splits
        output_file: the file where the splits should be output
    """
```

This outputs a `JSON` file which can be used by the benchmarking scripts to create 
shuffles.

## Benchmarking Models

### Without Data Splits

You might not care about train/test splits (e.g., because the test split is a actually 
a validation split and you have a completely different test set, where the images are
in a different project), you can always create your shuffles yourself using the 
DeepLabCut API. You can then modify the `pytorch_config.py` files for your shuffles if 
you want to modify the base configurations. (e.g., a different number of deconvolutional
layers).

Then, you can use the `benchmark_train.py` file to train models on your shuffles.

All you need to do is define your `RunParameters`, and call `main`. This will 
sequentially launch each training run. The first element is simply the shuffle which 
should be benchmarked. The next few elements of `RunParameters` describe which 
DeepLabCut API methods you want to call (train, evaluate, video analysis, ...). 
Then, a bunch of other parameters allow you to select exactly how your model should be 
trained and evaluated (batch size, ...).


```python
@dataclass
class RunParameters:
    """Parameters on what to run for each shuffle"""

    shuffle: Shuffle
    train: bool = False
    evaluate: bool = False
    analyze_videos: bool = False
    track: bool = False
    create_labeled_video: bool = False
    device: str = "cuda:0"
    train_params: TrainParameters | None = None
    detector_train_params: TrainParameters | None = None
    snapshot_path: Path | None = None
    detector_path: Path | None = None
    eval_params: EvalParameters | None = None
    video_analysis_params: VideoAnalysisParameters | None = None

    def __post_init__(self):
        if (
            self.analyze_videos is None or self.track or self.create_labeled_video
        ) and self.video_analysis_params is None:
            raise ValueError(f"Must specify video_analysis_params")


def main(runs: list[RunParameters]) -> None:
    """Runs benchmarking scripts for DeepLabCut

    Args:
        runs:
    """
    for run in runs:
        run.shuffle.project.update_iteration_in_config()

        if wandb.run is not None:  # TODO: Finish wandb run in DLC
            wandb.finish()

        print(f"Running {run.shuffle}")
        try:
            run_dlc(run)
        except Exception as err:
            print(f"Failed to run {run}: {err}")
            raise err
```

### With Data Splits

When benchmarking with data splits, a nice feature would be able to benchmark without
having to modify all `pytorch_config.yaml` files manually (as usually, you'll want to 
train exactly the same model architecture on different data splits). This can be done
with the `benchmark_run_experiments.py` script.

Here, you can define different variants of models with `ModelConfig` and train these
models on each one of your data splits. These parameters are made to customize
backbones and single animal heads, so they really can only be used for single animal or
top-down training. If you want to be able to easily update other parameters, you can 
either make your own classes or simply pass the updates as a dictionary.

There are also classes to update training parameters (batch size, epochs), augmentation,
optimizer, scheduler and logging parameters.
