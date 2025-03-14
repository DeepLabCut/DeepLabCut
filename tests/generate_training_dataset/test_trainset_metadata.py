#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests for deeplabcut/generate_training_dataset/metadata.py"""
from __future__ import annotations
import pickle

import pytest
from ruamel.yaml import YAML

import deeplabcut.generate_training_dataset.metadata as metadata
from deeplabcut.core.engine import Engine
from deeplabcut.utils import auxiliaryfunctions

SHUFFLE_DATA = [
    {"name": "pJun17-t50s1", "index": 1, "train_fraction": 0.5, "split": 1, "engine": "torch"},
    {"name": "pJun17-t50s2", "index": 2, "train_fraction": 0.5, "split": 1, "engine": "tf"},
    {"name": "pJun17-t60s1", "index": 1, "train_fraction": 0.6, "split": 2, "engine": "torch"},
    {"name": "pJun17-t60s2", "index": 2, "train_fraction": 0.6, "split": 3, "engine": "torch"},
]
SPLITS_DATA = {
    1: {"train": [0, 1], "test": [2, 3]},
    2: {"train": [0, 1, 2], "test": [3, 4]},
    3: {"train": [4, 3, 2], "test": [1, 0]},
}

BASE_SPLIT = metadata.DataSplit(train_indices=(1, 2), test_indices=(3, 4))
# Splits that should be equal to the base
EQ_SPLIT = metadata.DataSplit(train_indices=(1, 2), test_indices=(3, 4))
# Splits that should not be equal to the base
ADD_SPLIT = metadata.DataSplit(train_indices=(1, 2, 5), test_indices=(3, 4))
ADD_SPLIT2 = metadata.DataSplit(train_indices=(1, 2), test_indices=(3, 4, 5))
SUBS_SPLIT = metadata.DataSplit(train_indices=(1, 3), test_indices=(2, 4))
DEL_SPLIT = metadata.DataSplit(train_indices=(1,), test_indices=(3, 4))
DEL_SPLIT2 = metadata.DataSplit(train_indices=(1, 2), test_indices=(3,))

SHUFFLES = {
    1: metadata.ShuffleMetadata("pJun17-t50s1", 0.5, 1, Engine.PYTORCH, BASE_SPLIT),
    2: metadata.ShuffleMetadata("pJun17-t50s2", 0.5, 2, Engine.PYTORCH, ADD_SPLIT),
    3: metadata.ShuffleMetadata("pJun17-t50s3", 0.5, 3, Engine.TF, BASE_SPLIT),
    4: metadata.ShuffleMetadata("pJun17-t50s4", 0.5, 4, Engine.PYTORCH, DEL_SPLIT),
}


@pytest.mark.parametrize(
    "data",
    [
        {
            "shuffles": {SHUFFLE_DATA[idx]["name"]: SHUFFLE_DATA[idx] for idx in [0, 1, 2]},
            "splits": {idx: SPLITS_DATA[idx] for idx in [1, 2]},
        },
        {
            "shuffles": {SHUFFLE_DATA[idx]["name"]: SHUFFLE_DATA[idx] for idx in [0]},
            "splits": {idx: SPLITS_DATA[idx] for idx in [1, 2]},
        },
    ],
)
@pytest.mark.parametrize("load_splits", [True, False])
def test_load_metadata(tmpdir, data: dict, load_splits: bool):
    """Tests that loading the metadata from files doesn't fail"""
    # write data to tmp file
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    with open(meta_path, "w") as f:
        YAML().dump(data, f)

    print(cfg_path)
    print(meta_path)
    print(data["shuffles"])
    print(data["splits"])
    print()

    for name, s in data["shuffles"].items():
        split = data["splits"][s["split"]]
        train, test = split["train"], split["test"]
        _create_doc_data(
            cfg, trainset_dir, s["train_fraction"], s["index"], train, test
        )

    trainset_meta = metadata.TrainingDatasetMetadata.load(
        str(cfg_path), load_splits=load_splits
    )
    for s in trainset_meta.shuffles:
        print(s)

    assert len(data["shuffles"]) == len(trainset_meta.shuffles)

    for s in trainset_meta.shuffles:
        shuffle_in = data["shuffles"][s.name]
        split_idx = data["splits"][shuffle_in["split"]]
        assert s.train_fraction == shuffle_in["train_fraction"]
        assert s.engine == Engine(shuffle_in["engine"])
        if load_splits:
            assert s.split is not None
            assert s.split.train_indices == tuple(split_idx["train"])
            assert s.split.test_indices == tuple(split_idx["test"])
        else:
            assert s.split is None
            s_with_split = s.load_split(cfg, trainset_dir)
            assert s_with_split.split.train_indices == tuple(split_idx["train"])
            assert s_with_split.split.test_indices == tuple(split_idx["test"])


@pytest.mark.parametrize("data", [
    {
        "task": "ch",
        "date": "Aug1",
        "shuffles": (SHUFFLES[1], ),
        "expected": {
            "shuffles": {
                SHUFFLES[1].name: {
                    "index": 1, "train_fraction": 0.5, "split": 1, "engine": "pytorch"
                }
            },
        }
    },
    {
        "task": "t",
        "date": "Jan1",
        "shuffles": (SHUFFLES[1], SHUFFLES[3]),
        "expected": {
            "shuffles": {
                SHUFFLES[1].name: {
                    "index": 1, "train_fraction": 0.5, "split": 1, "engine": "pytorch"
                },
                SHUFFLES[3].name: {
                    "index": 3,
                    "train_fraction": 0.5,
                    "split": 1,
                    "engine": "tensorflow",
                },
            },
        }
    },
    {
        "task": "t",
        "date": "Jan1",
        "shuffles": (SHUFFLES[1], SHUFFLES[2]),
        "expected": {
            "shuffles": {
                SHUFFLES[1].name: {
                    "index": 1, "train_fraction": 0.5, "split": 1, "engine": "pytorch"
                },
                SHUFFLES[2].name: {
                    "index": 2, "train_fraction": 0.5, "split": 2, "engine": "pytorch"
                },
            },
        },
    },
    {
        "shuffles": (SHUFFLES[1], SHUFFLES[2], SHUFFLES[3]),
        "expected": {
            "shuffles": {
                SHUFFLES[1].name: {
                    "index": 1, "train_fraction": 0.5, "split": 1, "engine": "pytorch"
                },
                SHUFFLES[2].name: {
                    "index": 2, "train_fraction": 0.5, "split": 2, "engine": "pytorch"
                },
                SHUFFLES[3].name: {
                    "index": 3,
                    "train_fraction": 0.5,
                    "split": 1,
                    "engine": "tensorflow",
                },
            },
        },
    },
])
def test_save_metadata_simple(tmpdir, data):
    """Tests that saving the metadata creates the expected file"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    trainset_meta = metadata.TrainingDatasetMetadata(cfg, data["shuffles"])
    print(trainset_meta)

    trainset_meta.save()
    with open(meta_path, "r") as f:
        meta = YAML().load(f)
    print(data)
    print(meta)
    assert data["expected"] == meta


@pytest.mark.parametrize("shuffles", [
    [SHUFFLES[i] for i in indices]
    for indices in [[1], [1, 2], [1, 2, 3], [1, 2, 4], [1, 3, 4], [1, 2, 3, 4]]
])
def test_save_metadata(tmpdir, shuffles):
    """Tests that saving the metadata and reloading it leads to the same instance"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    for s in shuffles:
        train, test = s.split.train_indices, s.split.test_indices,
        _create_doc_data(cfg, trainset_dir, s.train_fraction, s.index, train, test)

    trainset_meta = metadata.TrainingDatasetMetadata(cfg, tuple(shuffles))
    print(trainset_meta)
    trainset_meta.save()
    reloaded = metadata.TrainingDatasetMetadata.load(cfg)
    print(reloaded)
    print()

    for s in trainset_meta.shuffles:
        print(s)
    print()
    for s in reloaded.shuffles:
        print(s)
    print()
    reloaded_with_splits = [s.load_split(cfg, trainset_dir) for s in reloaded.shuffles]
    assert len(reloaded.shuffles) == len(trainset_meta.shuffles)
    assert len(reloaded_with_splits) == len(trainset_meta.shuffles)
    assert tuple(reloaded_with_splits) == trainset_meta.shuffles


def test_add_shuffle(tmpdir):
    """Tests that a shuffle can be added correctlt"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    trainset_meta = metadata.TrainingDatasetMetadata(cfg, (SHUFFLES[1], ))
    trainset_meta_added = trainset_meta.add(SHUFFLES[2])
    assert len(trainset_meta.shuffles) == 1
    assert len(trainset_meta_added.shuffles) == 2
    assert trainset_meta_added.shuffles == (SHUFFLES[1], SHUFFLES[2])


def test_add_shuffle_twice(tmpdir):
    """Tests that a shuffle can be added correctlt"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    trainset_meta = metadata.TrainingDatasetMetadata(cfg, (SHUFFLES[1], ))
    trainset_meta_added = trainset_meta.add(SHUFFLES[2])
    trainset_meta_added_2 = trainset_meta.add(SHUFFLES[2])
    assert len(trainset_meta.shuffles) == 1
    assert trainset_meta.shuffles == (SHUFFLES[1], )
    assert len(trainset_meta_added.shuffles) == len(trainset_meta_added_2.shuffles)
    assert trainset_meta_added.shuffles == trainset_meta_added_2.shuffles


def test_add_shuffle_sorts_to_correct_order(tmpdir):
    """Tests that a shuffle can be added correctlt"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    trainset_meta = metadata.TrainingDatasetMetadata(cfg, (SHUFFLES[1], SHUFFLES[3]))
    trainset_meta_added = trainset_meta.add(SHUFFLES[2])
    assert len(trainset_meta.shuffles) == 2
    assert len(trainset_meta_added.shuffles) == 3
    assert trainset_meta_added.shuffles == (SHUFFLES[1], SHUFFLES[2], SHUFFLES[3])


@pytest.mark.parametrize("shuffles", [
    indices for indices in [[1], [1, 2], [1, 2, 3], [1, 2, 4], [1, 3, 4], [1, 2, 3, 4]]
])
@pytest.mark.parametrize("shuffle_to_add", [1, 2, 3, 4])
def test_add_shuffle(tmpdir, shuffles, shuffle_to_add):
    """Tests """
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    trainset_meta = metadata.TrainingDatasetMetadata(
        cfg, tuple([SHUFFLES[i] for i in shuffles])
    )
    if shuffle_to_add in shuffles:
        with pytest.raises(RuntimeError):
            trainset_meta_added = trainset_meta.add(
                SHUFFLES[shuffle_to_add], overwrite=False
            )

        trainset_meta_added = trainset_meta.add(
            SHUFFLES[shuffle_to_add], overwrite=True
        )
        assert len(trainset_meta_added.shuffles) == len(shuffles)
        assert [s.index for s in trainset_meta_added.shuffles] == shuffles
    else:
        trainset_meta_added = trainset_meta.add(
            SHUFFLES[shuffle_to_add], overwrite=False
        )
        indices = [s.index for s in trainset_meta_added.shuffles]
        assert len(trainset_meta_added.shuffles) == len(shuffles) + 1
        assert indices == list(sorted(shuffles + [shuffle_to_add]))


@pytest.mark.parametrize(
    "split1, split2, equal",
    [
        (BASE_SPLIT, EQ_SPLIT, True),
        (BASE_SPLIT, ADD_SPLIT, False),
        (BASE_SPLIT, ADD_SPLIT2, False),
        (BASE_SPLIT, SUBS_SPLIT, False),
        (BASE_SPLIT, DEL_SPLIT, False),
        (BASE_SPLIT, DEL_SPLIT2, False),
    ],
)
def test_data_split_equality(split1, split2, equal):
    """Tests that equality functions as expected for DataSplits"""
    print(split1)
    print(split2)
    print(equal)
    assert (split1 == split2) == equal


@pytest.mark.parametrize("split_idx", [1, 4, 20, 1000])
@pytest.mark.parametrize("indices", [(2, 1), (10, 1), (1, 21, 20), (1, 2, 4, 3)])
@pytest.mark.parametrize("sorted_indices", [(1, 2), (10, 12), (3, 4), (1, 1000, 1200)])
def test_data_split_requires_sorted(
    split_idx: int, indices: tuple[int], sorted_indices: tuple[int]
):
    """Tests that equality functions as expected for DataSplits"""
    with pytest.raises(RuntimeError):
        metadata.DataSplit(
            train_indices=tuple(indices), test_indices=tuple(sorted_indices)
        )

    with pytest.raises(RuntimeError):
        metadata.DataSplit(
            train_indices=tuple(sorted_indices), test_indices=tuple(indices)
        )

    with pytest.raises(RuntimeError):
        metadata.DataSplit(
            train_indices=tuple(indices), test_indices=tuple(indices)
        )

    metadata.DataSplit(
        train_indices=tuple(sorted_indices), test_indices=tuple(sorted_indices)
    )


@pytest.mark.parametrize("shuffles", [
    (
        {"idx": 3, "train": [1], "test": [2], "train_fraction": 0.5},
    ),
    (
        {"idx": 1, "train": [1], "test": [2], "train_fraction": 0.5},
        {"idx": 5, "train": [1, 2, 3], "test": [4, 5], "train_fraction": 0.6},
        {"idx": 4, "train": [1, 3], "test": [2], "train_fraction": 0.66},
    ),
])
def test_create_metadata_from_shuffles(tmpdir, shuffles):
    """Tests that equality functions as expected for DataSplits"""
    cfg, cfg_path, trainset_dir, meta_path = _create_project_with_config(tmpdir)
    print(trainset_dir)
    for s in shuffles:
        doc = f"Documentation_data-ex_{s['train_fraction']}shuffle{s['idx']}.pickle"
        doc_path = trainset_dir.join(doc)
        with open(doc_path, "wb") as f:
            pickle.dump(
                [[], s["train"], s["test"], s['train_fraction']], f,
                pickle.HIGHEST_PROTOCOL
            )

    trainset_metadata = metadata.TrainingDatasetMetadata.create(cfg)
    print()
    print(trainset_metadata)
    assert len(trainset_metadata.shuffles) == len(shuffles)

    for shuffle_data, shuffle in zip(shuffles, trainset_metadata.shuffles):
        print(shuffle.index)
        assert shuffle_data["idx"] == shuffle.index
        assert shuffle_data["train_fraction"] == shuffle.train_fraction
        assert tuple(shuffle_data["train"]) == shuffle.split.train_indices
        assert tuple(shuffle_data["test"]) == shuffle.split.test_indices
    print()


def _create_project_with_config(
    tmp,
    task: str = "example",
    date: str = "Feb21",
    scorer: str = "wayneRooney",
    iteration: int = 0,
    engine: str | None = None,
):
    project_dir = tmp.mkdir("ex-ample-2024-02-21")
    cfg = {
        "Task": task,
        "date": date,
        "scorer": scorer,
        "iteration": iteration,
        "project_path": str(project_dir),
    }
    if engine is not None:
        cfg["engine"] = engine

    cfg_path = project_dir.join("config.yaml")
    with open(cfg_path, "w") as file:
        YAML().dump(cfg, file)

    it = f"iteration-{iteration}"
    dir_name = "UnaugmentedDataSet_" + task + date
    trainset_dir = project_dir.mkdir("training-datasets").mkdir(it).mkdir(dir_name)

    meta_path = trainset_dir.join("metadata.yaml")
    return cfg, cfg_path, trainset_dir, meta_path


def _create_doc_data(
    cfg,
    trainset_dir,
    train_frac,
    shuffle,
    train_indices,
    test_indices,
) -> None:
    _, doc_path = auxiliaryfunctions.get_data_and_metadata_filenames(
        trainset_dir, train_frac, shuffle, cfg
    )
    auxiliaryfunctions.save_metadata(
        doc_path, {}, list(train_indices), list(test_indices), train_frac
    )
