import pytest
import deeplabcut.pose_estimation_pytorch.solvers.utils as deeplabcut_pytorch_pose_utils
import deeplabcut.utils.auxiliaryfunctions as deeplabcut_utils_auxiliary_functions

test_data = [
    ([
        "/path/to/snapshot-100.pt",
        "/path/to/snapshot-10.pt",
        "/path/to/snapshot-5.pt",
        "/path/to/snapshot-50.pt",
        "/path/to/snapshot5-50.pt",
        "/path/to/snapshot1-00.pt",
    ], [
        "/path/to/snapshot-100.pt", "/path/to/snapshot-10.pt",
        "/path/to/snapshot-5.pt", "/path/to/snapshot-50.pt"
    ]),
    ([
        "\\path\\to\\snapshot-100.pt",
        "\\path\\to\\snapshot-10.pt",
        "\\path\\to\\snapshot-5.pt",
        "\\path\\to\\snapshot-50.pt",
        "\\path\\to\\snapshot5-50.pt",
        "\\path\\to\\snapshot1-00.pt",
    ], [
        "\\path\\to\\snapshot-100.pt", "\\path\\to\\snapshot-10.pt",
        "\\path\\to\\snapshot-5.pt", "\\path\\to\\snapshot-50.pt"
    ]),
    ([
        "\path\to\snapshot-100.pt",
        "\path\to\snapshot-10.pt",
        "\path\to\snapshot-5.pt",
        "\path\to\snapshot-50.pt",
        "\path\to\snapshot5-50.pt",
        "\path\to\snapshot1-00.pt",
    ], [
        "\path\to\snapshot-100.pt", "\path\to\snapshot-10.pt",
        "\path\to\snapshot-5.pt", "\path\to\snapshot-50.pt"
    ]),
    ([
        "C:\\path\\to\\snapshot-100.pt",
        "C:\\path\\to\\snapshot-10.pt",
        "C:\\path\\to\\snapshot-5.pt",
        "C:\\path\\to\\snapshot-50.pt",
        "C:\\path\\to\\snapshot5-50.pt",
        "C:\\path\\to\\snapshot1-00.pt",
    ], [
        "C:\\path\\to\\snapshot-100.pt", "C:\\path\\to\\snapshot-10.pt",
        "C:\\path\\to\\snapshot-5.pt", "C:\\path\\to\\snapshot-50.pt"
    ]),
    ([
        "C:\path\to\snapshot-100.pt",
        "C:\path\to\snapshot-10.pt",
        "C:\path\to\snapshot-5.pt",
        "C:\path\to\snapshot-50.pt",
        "C:\path\to\snapshot5-50.pt",
        "C:\path\to\snapshot1-00.pt",
    ], [
        "C:\path\to\snapshot-100.pt", "C:\path\to\snapshot-10.pt",
        "C:\path\to\snapshot-5.pt", "C:\path\to\snapshot-50.pt"
    ]),
]


@pytest.mark.parametrize("paths,expected_verified_paths", test_data)
def test_verify_paths_model(paths, expected_verified_paths):
    with pytest.warns():
        verified_paths = deeplabcut_pytorch_pose_utils.verify_paths(paths)
        assert verified_paths == expected_verified_paths

test_data = [
    ([
        "/path/to/snapshot-100.pt",
        "/path/to/snapshot-10.pt",
        "/path/to/snapshot-5.pt",
        "/path/to/snapshot-50.pt",
        "/path/to/snapshot5-50.pt",
        "/path/to/snapshot1-00.pt",
    ], [
        "/path/to/snapshot-5.pt", "/path/to/snapshot-10.pt",
        "/path/to/snapshot-50.pt", "/path/to/snapshot-100.pt"
    ]),
    ([
        "\\path\\to\\snapshot-100.pt",
        "\\path\\to\\snapshot-10.pt",
        "\\path\\to\\snapshot-5.pt",
        "\\path\\to\\snapshot-50.pt",
        "\\path\\to\\snapshot5-50.pt",
        "\\path\\to\\snapshot1-00.pt",
    ], [
        "\\path\\to\\snapshot-5.pt", "\\path\\to\\snapshot-10.pt",
        "\\path\\to\\snapshot-50.pt", "\\path\\to\\snapshot-100.pt"
    ]),
    ([
        "\path\to\snapshot-100.pt",
        "\path\to\snapshot-10.pt",
        "\path\to\snapshot-5.pt",
        "\path\to\snapshot-50.pt",
        "\path\to\snapshot5-50.pt",
        "\path\to\snapshot1-00.pt",
    ], [
        "\path\to\snapshot-5.pt", "\path\to\snapshot-10.pt",
        "\path\to\snapshot-50.pt", "\path\to\snapshot-100.pt"
    ]),
    ([
        "C:\\path\\to\\snapshot-100.pt",
        "C:\\path\\to\\snapshot-10.pt",
        "C:\\path\\to\\snapshot-5.pt",
        "C:\\path\\to\\snapshot-50.pt",
        "C:\\path\\to\\snapshot5-50.pt",
        "C:\\path\\to\\snapshot1-00.pt",
    ], [
        "C:\\path\\to\\snapshot-5.pt", "C:\\path\\to\\snapshot-10.pt",
        "C:\\path\\to\\snapshot-50.pt", "C:\\path\\to\\snapshot-100.pt"
    ]),
    ([
        "C:\path\to\snapshot-100.pt",
        "C:\path\to\snapshot-10.pt",
        "C:\path\to\snapshot-5.pt",
        "C:\path\to\snapshot-50.pt",
        "C:\path\to\snapshot5-50.pt",
        "C:\path\to\snapshot1-00.pt",
    ], [
        "C:\path\to\snapshot-5.pt", "C:\path\to\snapshot-10.pt",
        "C:\path\to\snapshot-50.pt", "C:\path\to\snapshot-100.pt"
    ]),
]


@pytest.mark.parametrize("paths,expected_sorted_paths", test_data)
def test_sort_model_paths(paths, expected_sorted_paths):
    with pytest.warns():
        sorted_paths = deeplabcut_pytorch_pose_utils.sort_paths(paths)
        assert sorted_paths == expected_sorted_paths


test_data = [([
    "/path/to/detector-snapshot-100.pt",
    "/path/to/detector-snapshot-10.pt",
    "/path/to/detector-snapshot-5.pt",
    "/path/to/detector-snapshot-50.pt",
    "/path/to/detector-snapshot5-50.pt",
    "/path/to/snapshot1-00.pt",
], [
    "/path/to/detector-snapshot-100.pt", "/path/to/detector-snapshot-10.pt",
    "/path/to/detector-snapshot-5.pt", "/path/to/detector-snapshot-50.pt"
]),
             ([
                 "\\path\\to\\detector-snapshot-100.pt",
                 "\\path\\to\\detector-snapshot-10.pt",
                 "\\path\\to\\detector-snapshot-5.pt",
                 "\\path\\to\\detector-snapshot-50.pt",
                 "\\path\\to\\detector-snapshot5-50.pt",
                 "\\path\\to\\detector-snapshot1-00.pt",
             ], [
                 "\\path\\to\\detector-snapshot-100.pt",
                 "\\path\\to\\detector-snapshot-10.pt",
                 "\\path\\to\\detector-snapshot-5.pt",
                 "\\path\\to\\detector-snapshot-50.pt"
             ]),
             ([
                 "\path\to\detector-snapshot-100.pt",
                 "\path\to\detector-snapshot-10.pt",
                 "\path\to\detector-snapshot-5.pt",
                 "\path\to\detector-snapshot-50.pt",
                 "\path\to\detector-snapshot5-50.pt",
                 "\path\to\snapshot1-00.pt",
             ], [
                 "\path\to\detector-snapshot-100.pt",
                 "\path\to\detector-snapshot-10.pt",
                 "\path\to\detector-snapshot-5.pt",
                 "\path\to\detector-snapshot-50.pt"
             ]),
             ([
                 "C:\\path\\to\\detector-snapshot-100.pt",
                 "C:\\path\\to\\detector-snapshot-10.pt",
                 "C:\\path\\to\\detector-snapshot-5.pt",
                 "C:\\path\\to\\detector-snapshot-50.pt",
                 "C:\\path\\to\\detector-snapshot5-50.pt",
                 "C:\\path\\to\\detector-snapshot1-00.pt",
             ], [
                 "C:\\path\\to\\detector-snapshot-100.pt",
                 "C:\\path\\to\\detector-snapshot-10.pt",
                 "C:\\path\\to\\detector-snapshot-5.pt",
                 "C:\\path\\to\\detector-snapshot-50.pt"
             ]),
             ([
                 "C:\path\to\detector-snapshot-100.pt",
                 "C:\path\to\detector-snapshot-10.pt",
                 "C:\path\to\detector-snapshot-5.pt",
                 "C:\path\to\detector-snapshot-50.pt",
                 "C:\path\to\detector-snapshot5-50.pt",
                 "C:\path\to\snapshot1-00.pt",
             ], [
                 "C:\path\to\detector-snapshot-100.pt",
                 "C:\path\to\detector-snapshot-10.pt",
                 "C:\path\to\detector-snapshot-5.pt",
                 "C:\path\to\detector-snapshot-50.pt"
             ])]


@pytest.mark.parametrize("paths,expected_verified_paths", test_data)
def test_verify_paths_detector(paths, expected_verified_paths):
    with pytest.warns():
        verified_paths = deeplabcut_pytorch_pose_utils.verify_paths(
            paths, r"^(.*)?detector-snapshot-(\d+)\.pt$")
        assert verified_paths == expected_verified_paths


test_data = [
    ([
        "/path/to/detector-snapshot-100.pt",
        "/path/to/detector-snapshot-10.pt",
        "/path/to/detector-snapshot-5.pt",
        "/path/to/detector-snapshot-50.pt",
        "/path/to/detector-snapshot5-50.pt",
        "/path/to/snapshot1-00.pt",
    ], [
        "/path/to/detector-snapshot-5.pt", "/path/to/detector-snapshot-10.pt",
        "/path/to/detector-snapshot-50.pt", "/path/to/detector-snapshot-100.pt"
    ]),
    ([
        "\\path\\to\\detector-snapshot-100.pt",
        "\\path\\to\\detector-snapshot-10.pt",
        "\\path\\to\\detector-snapshot-5.pt",
        "\\path\\to\\detector-snapshot-50.pt",
        "\\path\\to\\detector-snapshot5-50.pt",
        "\\path\\to\\detector-snapshot1-00.pt",
    ], [
        "\\path\\to\\detector-snapshot-5.pt",
        "\\path\\to\\detector-snapshot-10.pt",
        "\\path\\to\\detector-snapshot-50.pt",
        "\\path\\to\\detector-snapshot-100.pt"
    ]),
    ([
        "\path\to\detector-snapshot-100.pt",
        "\path\to\detector-snapshot-10.pt",
        "\path\to\detector-snapshot-5.pt",
        "\path\to\detector-snapshot-50.pt",
        "\path\to\detector-snapshot5-50.pt",
        "\path\to\snapshot1-00.pt",
    ], [
        "\path\to\detector-snapshot-5.pt", "\path\to\detector-snapshot-10.pt",
        "\path\to\detector-snapshot-50.pt", "\path\to\detector-snapshot-100.pt"
    ]),
    ([
        "C:\\path\\to\\detector-snapshot-100.pt",
        "C:\\path\\to\\detector-snapshot-10.pt",
        "C:\\path\\to\\detector-snapshot-5.pt",
        "C:\\path\\to\\detector-snapshot-50.pt",
        "C:\\path\\to\\detector-snapshot5-50.pt",
        "C:\\path\\to\\detector-snapshot1-00.pt",
    ], [
        "C:\\path\\to\\detector-snapshot-5.pt",
        "C:\\path\\to\\detector-snapshot-10.pt",
        "C:\\path\\to\\detector-snapshot-50.pt",
        "C:\\path\\to\\detector-snapshot-100.pt"
    ]),
    ([
        "C:\path\to\detector-snapshot-100.pt",
        "C:\path\to\detector-snapshot-10.pt",
        "C:\path\to\detector-snapshot-5.pt",
        "C:\path\to\detector-snapshot-50.pt",
        "C:\path\to\detector-snapshot5-50.pt",
        "C:\path\to\snapshot1-00.pt",
    ], [
        "C:\path\to\detector-snapshot-5.pt",
        "C:\path\to\detector-snapshot-10.pt",
        "C:\path\to\detector-snapshot-50.pt",
        "C:\path\to\detector-snapshot-100.pt"
    ])
]


@pytest.mark.parametrize("paths,expected_sorted_paths", test_data)
def test_sort_detector_paths(paths, expected_sorted_paths):
    with pytest.warns():
        sorted_paths = deeplabcut_pytorch_pose_utils.sort_paths(
            paths, r"^(.*)?detector-snapshot-(\d+)\.pt$")
        assert sorted_paths == expected_sorted_paths
