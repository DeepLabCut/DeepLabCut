# Ruff manual-fix report

Generated from: `.`

Total remaining issues: **1437**

## Summary

| Rule | Count | Note |
|---|---:|---|
| `E501` | 333 | Line too long. Prefer wrapping expressions, splitting long strings/comments, or extracting variables. |
| `F401` | 331 | Unused import. Usually safe to delete; verify imports with side effects. |
| `B905` | 176 |  |
| `F841` | 141 |  |
| `E402` | 93 | Module import not at top of file. Move imports above executable code if possible. |
| `UP031` | 76 | Old `%` formatting. Convert to f-strings or `.format()` where appropriate. |
| `B007` | 51 | Unused loop variable. Rename to `_` or use it. |
| `B028` | 49 |  |
| `F403` | 36 | `from x import *` makes names unclear. Replace with explicit imports. |
| `E712` | 22 |  |
| `F821` | 22 | Undefined name. Usually a real bug or missing import. |
| `B904` | 19 | Inside `except`, use `raise ... from e` to preserve exception chaining. |
| `E722` | 19 | Bare `except:`. Catch `Exception` or a narrower exception type. |
| `F405` | 16 | Likely consequence of `import *`. Import the name explicitly. |
| `E721` | 14 | Avoid direct `type(x) == Y`; prefer `isinstance(x, Y)`. |
| `B006` | 12 |  |
| `E711` | 7 |  |
| `E731` | 4 |  |
| `B008` | 3 | Function call in default arg. Use `None` + initialize inside the function. |
| `B023` | 2 | Function closes over loop variable. Bind it via default arg or helper. |
| `B024` | 2 | ABC without abstract method. Add `@abstractmethod` or remove ABC intent. |
| `F811` | 2 | Redefined while unused. Remove duplicate or rename. |
| `B011` | 1 |  |
| `B012` | 1 | Jump statement in `finally` can swallow exceptions. Restructure flow. |
| `B016` | 1 | Raise an exception instance/class, not a literal. |
| `B017` | 1 | Use a more specific exception with `assertRaises`. |
| `B020` | 1 | Loop variable overrides iterator. Rename loop variables. |
| `B027` | 1 | Empty method in ABC without abstract decorator. Add `@abstractmethod` or implement it. |
| `UP028` | 1 |  |

## Suggested triage order

1. `F403` — `from x import *` makes names unclear. Replace with explicit imports.
2. `F405` — Likely consequence of `import *`. Import the name explicitly.
3. `F821` — Undefined name. Usually a real bug or missing import.
4. `E722` — Bare `except:`. Catch `Exception` or a narrower exception type.
5. `B904` — Inside `except`, use `raise ... from e` to preserve exception chaining.
6. `E402` — Module import not at top of file. Move imports above executable code if possible.
7. `F401` — Unused import. Usually safe to delete; verify imports with side effects.
8. `E501` — Line too long. Prefer wrapping expressions, splitting long strings/comments, or extracting variables.

## Table of contents by rule

- [E501 (333)](#e501)
- [F401 (331)](#f401)
- [B905 (176)](#b905)
- [F841 (141)](#f841)
- [E402 (93)](#e402)
- [UP031 (76)](#up031)
- [B007 (51)](#b007)
- [B028 (49)](#b028)
- [F403 (36)](#f403)
- [E712 (22)](#e712)
- [F821 (22)](#f821)
- [B904 (19)](#b904)
- [E722 (19)](#e722)
- [F405 (16)](#f405)
- [E721 (14)](#e721)
- [B006 (12)](#b006)
- [E711 (7)](#e711)
- [E731 (4)](#e731)
- [B008 (3)](#b008)
- [B023 (2)](#b023)
- [B024 (2)](#b024)
- [F811 (2)](#f811)
- [B011 (1)](#b011)
- [B012 (1)](#b012)
- [B016 (1)](#b016)
- [B017 (1)](#b017)
- [B020 (1)](#b020)
- [B027 (1)](#b027)
- [UP028 (1)](#up028)

## E501

Count: **333**
Hint: Line too long. Prefer wrapping expressions, splitting long strings/comments, or extracting variables.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 42 |
| `deeplabcut\cli.py` | 30 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 19 |
| `deeplabcut\compat.py` | 16 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 15 |
| `deeplabcut\create_project\modelzoo.py` | 13 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 13 |
| `deeplabcut\pose_estimation_3d\plotting3D.py` | 11 |
| `deeplabcut\utils\conversioncode.py` | 11 |
| `deeplabcut\utils\frameselectiontools.py` | 10 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 9 |
| `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` | 9 |
| `deeplabcut\utils\auxfun_videos.py` | 9 |
| `deeplabcut\benchmark\benchmarks.py` | 8 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 8 |
| `deeplabcut\refine_training_dataset\outlier_frames.py` | 7 |
| `deeplabcut\utils\auxiliaryfunctions.py` | 6 |
| `docs\recipes\flip_and_rotate.ipynb` | 6 |
| `deeplabcut\utils\auxfun_multianimal.py` | 5 |
| `deeplabcut\create_project\add.py` | 3 |
| `deeplabcut\create_project\new.py` | 3 |
| `deeplabcut\create_project\new_3d.py` | 3 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 3 |
| `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` | 3 |
| `deeplabcut\gui\tracklet_toolbox.py` | 3 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` | 3 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 3 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\spatiotemporal_adapt.py` | 3 |
| `deeplabcut\refine_training_dataset\stitch.py` | 3 |
| `deeplabcut\utils\make_labeled_video.py` | 3 |
| `deeplabcut\gui\window.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` | 2 |
| `deeplabcut\modelzoo\utils.py` | 2 |
| `deeplabcut\modelzoo\video_inference.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\config\make_pose_config.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\core\train_multianimal.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\training.py` | 2 |
| `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` | 2 |
| `deeplabcut\pose_tracking_pytorch\processor\processor.py` | 2 |
| `deeplabcut\utils\auxfun_models.py` | 2 |
| `deeplabcut\utils\auxiliaryfunctions_3d.py` | 2 |
| `deeplabcut\utils\pseudo_label.py` | 2 |
| `tests\pose_estimation_pytorch\other\test_match_predictions_to_gt.py` | 2 |
| `testscript_cli.py` | 2 |
| `deeplabcut\__main__.py` | 1 |
| `deeplabcut\gui\tabs\extract_outlier_frames.py` | 1 |
| `deeplabcut\gui\tabs\modelzoo.py` | 1 |
| `deeplabcut\gui\tabs\refine_tracklets.py` | 1 |
| `deeplabcut\gui\tabs\train_network.py` | 1 |
| `deeplabcut\gui\widgets.py` | 1 |
| `deeplabcut\modelzoo\weight_initialization.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\modules\conv_block.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\post_processing\match_predictions_to_gt.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\train.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` | 1 |
| `deeplabcut\post_processing\analyze_skeleton.py` | 1 |
| `deeplabcut\utils\visualization.py` | 1 |
| `examples\JUPYTER\Demo_yourowndata.ipynb` | 1 |
| `examples\testscript_3d.py` | 1 |
| `examples\testscript_deterministicwithResNet152.py` | 1 |
| `examples\testscript_mobilenets.py` | 1 |
| `examples\testscript_openfielddata.py` | 1 |
| `examples\testscript_pretrained_models.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (42)

| Line | Col | Message |
|---:|---:|---|
| 76 | 121 | Line too long (173 > 120) |
| 115 | 121 | Line too long (235 > 120) |
| 151 | 121 | Line too long (150 > 120) |
| 165 | 121 | Line too long (136 > 120) |
| 226 | 121 | Line too long (205 > 120) |
| 230 | 121 | Line too long (138 > 120) |
| 233 | 121 | Line too long (205 > 120) |
| 334 | 121 | Line too long (153 > 120) |
| 335 | 121 | Line too long (155 > 120) |
| 336 | 121 | Line too long (150 > 120) |
| 337 | 121 | Line too long (145 > 120) |
| 499 | 121 | Line too long (235 > 120) |
| 535 | 121 | Line too long (150 > 120) |
| 549 | 121 | Line too long (136 > 120) |
| 643 | 121 | Line too long (184 > 120) |
| 646 | 121 | Line too long (205 > 120) |
| 650 | 121 | Line too long (138 > 120) |
| 653 | 121 | Line too long (205 > 120) |
| 663 | 121 | Line too long (129 > 120) |
| 902 | 121 | Line too long (143 > 120) |
| 1077 | 121 | Line too long (133 > 120) |
| 1152 | 121 | Line too long (126 > 120) |
| 1154 | 121 | Line too long (142 > 120) |
| 1155 | 121 | Line too long (145 > 120) |
| 1156 | 121 | Line too long (146 > 120) |
| 1157 | 121 | Line too long (128 > 120) |
| 1168 | 121 | Line too long (122 > 120) |
| 1174 | 121 | Line too long (136 > 120) |
| 1176 | 121 | Line too long (140 > 120) |
| 1180 | 121 | Line too long (123 > 120) |
| 1185 | 121 | Line too long (121 > 120) |
| 1188 | 121 | Line too long (122 > 120) |
| 1219 | 121 | Line too long (235 > 120) |
| 1460 | 121 | Line too long (155 > 120) |
| 1463 | 121 | Line too long (140 > 120) |
| 1470 | 121 | Line too long (136 > 120) |
| 1476 | 121 | Line too long (133 > 120) |
| 1512 | 121 | Line too long (146 > 120) |
| 1515 | 121 | Line too long (165 > 120) |
| 1573 | 121 | Line too long (235 > 120) |
| 1626 | 121 | Line too long (123 > 120) |
| 1741 | 121 | Line too long (193 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:76"
```

#### `deeplabcut\cli.py` (30)

| Line | Col | Message |
|---:|---:|---|
| 52 | 121 | Line too long (187 > 120) |
| 63 | 121 | Line too long (139 > 120) |
| 65 | 121 | Line too long (177 > 120) |
| 70 | 121 | Line too long (149 > 120) |
| 73 | 121 | Line too long (158 > 120) |
| 76 | 121 | Line too long (161 > 120) |
| 143 | 121 | Line too long (132 > 120) |
| 144 | 121 | Line too long (132 > 120) |
| 156 | 121 | Line too long (132 > 120) |
| 161 | 121 | Line too long (193 > 120) |
| 175 | 121 | Line too long (136 > 120) |
| 190 | 121 | Line too long (169 > 120) |
| 208 | 121 | Line too long (148 > 120) |
| 334 | 121 | Line too long (180 > 120) |
| 341 | 121 | Line too long (140 > 120) |
| 342 | 121 | Line too long (158 > 120) |
| 343 | 121 | Line too long (129 > 120) |
| 344 | 121 | Line too long (144 > 120) |
| 353 | 121 | Line too long (133 > 120) |
| 361 | 121 | Line too long (127 > 120) |
| 362 | 121 | Line too long (150 > 120) |
| 369 | 121 | Line too long (152 > 120) |
| 400 | 121 | Line too long (130 > 120) |
| 406 | 121 | Line too long (128 > 120) |
| 419 | 121 | Line too long (134 > 120) |
| 422 | 121 | Line too long (165 > 120) |
| 425 | 121 | Line too long (175 > 120) |
| 439 | 121 | Line too long (127 > 120) |
| 537 | 121 | Line too long (132 > 120) |
| 609 | 121 | Line too long (145 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\cli.py:52"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (19)

| Line | Col | Message |
|---:|---:|---|
| 41 | 121 | Line too long (123 > 120) |
| 68 | 121 | Line too long (151 > 120) |
| 74 | 121 | Line too long (177 > 120) |
| 158 | 121 | Line too long (139 > 120) |
| 503 | 121 | Line too long (127 > 120) |
| 516 | 121 | Line too long (311 > 120) |
| 525 | 121 | Line too long (130 > 120) |
| 562 | 121 | Line too long (136 > 120) |
| 623 | 121 | Line too long (138 > 120) |
| 626 | 121 | Line too long (143 > 120) |
| 635 | 121 | Line too long (137 > 120) |
| 636 | 121 | Line too long (138 > 120) |
| 645 | 121 | Line too long (125 > 120) |
| 652 | 121 | Line too long (161 > 120) |
| 653 | 121 | Line too long (146 > 120) |
| 1062 | 121 | Line too long (131 > 120) |
| 1066 | 121 | Line too long (141 > 120) |
| 1121 | 121 | Line too long (164 > 120) |
| 1125 | 121 | Line too long (144 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:41"
```

#### `deeplabcut\compat.py` (16)

| Line | Col | Message |
|---:|---:|---|
| 587 | 121 | Line too long (176 > 120) |
| 593 | 121 | Line too long (146 > 120) |
| 602 | 121 | Line too long (141 > 120) |
| 609 | 121 | Line too long (137 > 120) |
| 610 | 121 | Line too long (141 > 120) |
| 611 | 121 | Line too long (147 > 120) |
| 612 | 121 | Line too long (149 > 120) |
| 1427 | 121 | Line too long (155 > 120) |
| 1430 | 121 | Line too long (140 > 120) |
| 1437 | 121 | Line too long (136 > 120) |
| 1443 | 121 | Line too long (133 > 120) |
| 1598 | 121 | Line too long (137 > 120) |
| 1599 | 121 | Line too long (141 > 120) |
| 1600 | 121 | Line too long (147 > 120) |
| 1601 | 121 | Line too long (149 > 120) |
| 1737 | 121 | Line too long (141 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\compat.py:587"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (15)

| Line | Col | Message |
|---:|---:|---|
| 53 | 121 | Line too long (141 > 120) |
| 204 | 121 | Line too long (176 > 120) |
| 207 | 121 | Line too long (146 > 120) |
| 216 | 121 | Line too long (141 > 120) |
| 223 | 121 | Line too long (137 > 120) |
| 224 | 121 | Line too long (141 > 120) |
| 225 | 121 | Line too long (147 > 120) |
| 226 | 121 | Line too long (149 > 120) |
| 249 | 121 | Line too long (139 > 120) |
| 350 | 121 | Line too long (178 > 120) |
| 879 | 121 | Line too long (131 > 120) |
| 883 | 121 | Line too long (130 > 120) |
| 915 | 121 | Line too long (154 > 120) |
| 935 | 121 | Line too long (262 > 120) |
| 938 | 121 | Line too long (140 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:53"
```

#### `deeplabcut\create_project\modelzoo.py` (13)

| Line | Col | Message |
|---:|---:|---|
| 102 | 121 | Line too long (160 > 120) |
| 206 | 121 | Line too long (127 > 120) |
| 209 | 121 | Line too long (148 > 120) |
| 212 | 121 | Line too long (182 > 120) |
| 336 | 121 | Line too long (135 > 120) |
| 339 | 121 | Line too long (156 > 120) |
| 342 | 121 | Line too long (190 > 120) |
| 361 | 121 | Line too long (138 > 120) |
| 366 | 121 | Line too long (151 > 120) |
| 534 | 121 | Line too long (138 > 120) |
| 537 | 121 | Line too long (159 > 120) |
| 540 | 121 | Line too long (193 > 120) |
| 647 | 121 | Line too long (126 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\modelzoo.py:102"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (13)

| Line | Col | Message |
|---:|---:|---|
| 29 | 121 | Line too long (184 > 120) |
| 31 | 121 | Line too long (151 > 120) |
| 33 | 121 | Line too long (172 > 120) |
| 34 | 121 | Line too long (152 > 120) |
| 36 | 121 | Line too long (132 > 120) |
| 51 | 121 | Line too long (127 > 120) |
| 52 | 121 | Line too long (121 > 120) |
| 55 | 121 | Line too long (155 > 120) |
| 119 | 121 | Line too long (166 > 120) |
| 159 | 121 | Line too long (226 > 120) |
| 264 | 121 | Line too long (317 > 120) |
| 271 | 121 | Line too long (146 > 120) |
| 286 | 121 | Line too long (157 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:29"
```

#### `deeplabcut\pose_estimation_3d\plotting3D.py` (11)

| Line | Col | Message |
|---:|---:|---|
| 84 | 121 | Line too long (159 > 120) |
| 87 | 121 | Line too long (287 > 120) |
| 93 | 121 | Line too long (159 > 120) |
| 99 | 121 | Line too long (140 > 120) |
| 103 | 121 | Line too long (141 > 120) |
| 106 | 121 | Line too long (216 > 120) |
| 109 | 121 | Line too long (216 > 120) |
| 112 | 121 | Line too long (216 > 120) |
| 115 | 121 | Line too long (219 > 120) |
| 130 | 121 | Line too long (148 > 120) |
| 155 | 121 | Line too long (243 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\plotting3D.py:84"
```

#### `deeplabcut\utils\conversioncode.py` (11)

| Line | Col | Message |
|---:|---:|---|
| 35 | 121 | Line too long (129 > 120) |
| 42 | 121 | Line too long (139 > 120) |
| 45 | 121 | Line too long (134 > 120) |
| 53 | 121 | Line too long (138 > 120) |
| 97 | 121 | Line too long (155 > 120) |
| 108 | 121 | Line too long (139 > 120) |
| 216 | 121 | Line too long (181 > 120) |
| 217 | 121 | Line too long (137 > 120) |
| 218 | 121 | Line too long (137 > 120) |
| 233 | 121 | Line too long (131 > 120) |
| 274 | 121 | Line too long (131 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\conversioncode.py:35"
```

#### `deeplabcut\utils\frameselectiontools.py` (10)

| Line | Col | Message |
|---:|---:|---|
| 32 | 121 | Line too long (125 > 120) |
| 75 | 121 | Line too long (125 > 120) |
| 124 | 121 | Line too long (125 > 120) |
| 125 | 121 | Line too long (126 > 120) |
| 128 | 121 | Line too long (130 > 120) |
| 172 | 121 | Line too long (128 > 120) |
| 214 | 121 | Line too long (125 > 120) |
| 215 | 121 | Line too long (126 > 120) |
| 218 | 121 | Line too long (130 > 120) |
| 222 | 121 | Line too long (140 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\frameselectiontools.py:32"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (9)

| Line | Col | Message |
|---:|---:|---|
| 49 | 121 | Line too long (140 > 120) |
| 77 | 121 | Line too long (220 > 120) |
| 85 | 121 | Line too long (268 > 120) |
| 116 | 121 | Line too long (124 > 120) |
| 188 | 121 | Line too long (126 > 120) |
| 298 | 121 | Line too long (201 > 120) |
| 304 | 121 | Line too long (177 > 120) |
| 501 | 121 | Line too long (165 > 120) |
| 509 | 121 | Line too long (130 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:49"
```

#### `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` (9)

| Line | Col | Message |
|---:|---:|---|
| 36 | 121 | Line too long (149 > 120) |
| 45 | 121 | Line too long (141 > 120) |
| 49 | 121 | Line too long (137 > 120) |
| 50 | 121 | Line too long (141 > 120) |
| 51 | 121 | Line too long (147 > 120) |
| 52 | 121 | Line too long (149 > 120) |
| 183 | 121 | Line too long (121 > 120) |
| 184 | 121 | Line too long (162 > 120) |
| 286 | 121 | Line too long (141 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\visualizemaps.py:36"
```

#### `deeplabcut\utils\auxfun_videos.py` (9)

| Line | Col | Message |
|---:|---:|---|
| 410 | 121 | Line too long (139 > 120) |
| 412 | 121 | Line too long (124 > 120) |
| 465 | 121 | Line too long (127 > 120) |
| 467 | 121 | Line too long (167 > 120) |
| 496 | 121 | Line too long (121 > 120) |
| 533 | 121 | Line too long (151 > 120) |
| 535 | 121 | Line too long (139 > 120) |
| 574 | 121 | Line too long (132 > 120) |
| 603 | 121 | Line too long (223 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_videos.py:410"
```

#### `deeplabcut\benchmark\benchmarks.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 27 | 121 | Line too long (776 > 120) |
| 29 | 121 | Line too long (149 > 120) |
| 55 | 121 | Line too long (1440 > 120) |
| 57 | 121 | Line too long (149 > 120) |
| 106 | 121 | Line too long (964 > 120) |
| 108 | 121 | Line too long (149 > 120) |
| 137 | 121 | Line too long (981 > 120) |
| 139 | 121 | Line too long (149 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\benchmark\benchmarks.py:27"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 441 | 121 | Line too long (127 > 120) |
| 443 | 121 | Line too long (142 > 120) |
| 444 | 121 | Line too long (145 > 120) |
| 450 | 121 | Line too long (155 > 120) |
| 453 | 121 | Line too long (242 > 120) |
| 455 | 121 | Line too long (237 > 120) |
| 458 | 121 | Line too long (164 > 120) |
| 461 | 121 | Line too long (180 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:441"
```

#### `deeplabcut\refine_training_dataset\outlier_frames.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 267 | 121 | Line too long (163 > 120) |
| 477 | 121 | Line too long (131 > 120) |
| 563 | 121 | Line too long (134 > 120) |
| 574 | 121 | Line too long (140 > 120) |
| 598 | 121 | Line too long (134 > 120) |
| 798 | 121 | Line too long (128 > 120) |
| 840 | 121 | Line too long (124 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\outlier_frames.py:267"
```

#### `deeplabcut\utils\auxiliaryfunctions.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 234 | 121 | Line too long (147 > 120) |
| 405 | 121 | Line too long (135 > 120) |
| 408 | 121 | Line too long (122 > 120) |
| 684 | 121 | Line too long (161 > 120) |
| 789 | 121 | Line too long (123 > 120) |
| 790 | 121 | Line too long (149 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions.py:234"
```

#### `docs\recipes\flip_and_rotate.ipynb` (6)

| Line | Col | Message |
|---:|---:|---|
| 10 | 121 | Line too long (155 > 120) |
| 19 | 121 | Line too long (155 > 120) |
| 19 | 121 | Line too long (155 > 120) |
| 20 | 121 | Line too long (155 > 120) |
| 20 | 121 | Line too long (155 > 120) |
| 22 | 121 | Line too long (155 > 120) |

Quick open commands:

```powershell
code -g "docs\recipes\flip_and_rotate.ipynb:10"
```

#### `deeplabcut\utils\auxfun_multianimal.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 94 | 121 | Line too long (148 > 120) |
| 122 | 121 | Line too long (136 > 120) |
| 242 | 121 | Line too long (161 > 120) |
| 243 | 121 | Line too long (157 > 120) |
| 358 | 121 | Line too long (136 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_multianimal.py:94"
```

#### `deeplabcut\create_project\add.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 39 | 121 | Line too long (122 > 120) |
| 42 | 121 | Line too long (163 > 120) |
| 45 | 121 | Line too long (203 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\add.py:39"
```

#### `deeplabcut\create_project\new.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 149 | 121 | Line too long (149 > 120) |
| 217 | 121 | Line too long (141 > 120) |
| 306 | 121 | Line too long (390 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new.py:149"
```

#### `deeplabcut\create_project\new_3d.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 35 | 121 | Line too long (140 > 120) |
| 91 | 121 | Line too long (123 > 120) |
| 126 | 121 | Line too long (295 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new_3d.py:35"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 425 | 121 | Line too long (171 > 120) |
| 451 | 121 | Line too long (142 > 120) |
| 544 | 121 | Line too long (163 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:425"
```

#### `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 261 | 121 | Line too long (128 > 120) |
| 262 | 121 | Line too long (213 > 120) |
| 268 | 121 | Line too long (210 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py:261"
```

#### `deeplabcut\gui\tracklet_toolbox.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 85 | 121 | Line too long (180 > 120) |
| 909 | 121 | Line too long (126 > 120) |
| 913 | 121 | Line too long (127 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tracklet_toolbox.py:85"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 34 | 121 | Line too long (127 > 120) |
| 47 | 121 | Line too long (311 > 120) |
| 56 | 121 | Line too long (130 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py:34"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 181 | 121 | Line too long (201 > 120) |
| 320 | 121 | Line too long (167 > 120) |
| 532 | 121 | Line too long (167 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:181"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 34 | 121 | Line too long (127 > 120) |
| 47 | 121 | Line too long (311 > 120) |
| 56 | 121 | Line too long (130 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py:34"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\spatiotemporal_adapt.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 53 | 121 | Line too long (167 > 120) |
| 55 | 121 | Line too long (182 > 120) |
| 57 | 121 | Line too long (169 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\spatiotemporal_adapt.py:53"
```

#### `deeplabcut\refine_training_dataset\stitch.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 1002 | 121 | Line too long (155 > 120) |
| 1005 | 121 | Line too long (140 > 120) |
| 1012 | 121 | Line too long (136 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\stitch.py:1002"
```

#### `deeplabcut\utils\make_labeled_video.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 1168 | 121 | Line too long (140 > 120) |
| 1175 | 121 | Line too long (136 > 120) |
| 1180 | 121 | Line too long (124 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\make_labeled_video.py:1168"
```

#### `deeplabcut\gui\window.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 372 | 121 | Line too long (312 > 120) |
| 554 | 121 | Line too long (141 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\window.py:372"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 85 | 121 | Line too long (125 > 120) |
| 130 | 121 | Line too long (191 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py:85"
```

#### `deeplabcut\modelzoo\utils.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 201 | 121 | Line too long (134 > 120) |
| 208 | 121 | Line too long (134 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\utils.py:201"
```

#### `deeplabcut\modelzoo\video_inference.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 480 | 121 | Line too long (122 > 120) |
| 549 | 121 | Line too long (134 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\video_inference.py:480"
```

#### `deeplabcut\pose_estimation_pytorch\config\make_pose_config.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 78 | 121 | Line too long (132 > 120) |
| 79 | 121 | Line too long (217 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\config\make_pose_config.py:78"
```

#### `deeplabcut\pose_estimation_tensorflow\core\train_multianimal.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 92 | 121 | Line too long (122 > 120) |
| 208 | 121 | Line too long (123 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\train_multianimal.py:92"
```

#### `deeplabcut\pose_estimation_tensorflow\training.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 28 | 121 | Line too long (136 > 120) |
| 176 | 121 | Line too long (161 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\training.py:28"
```

#### `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 332 | 121 | Line too long (121 > 120) |
| 346 | 121 | Line too long (126 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py:332"
```

#### `deeplabcut\pose_tracking_pytorch\processor\processor.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 131 | 121 | Line too long (154 > 120) |
| 143 | 121 | Line too long (143 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\processor\processor.py:131"
```

#### `deeplabcut\utils\auxfun_models.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 50 | 121 | Line too long (167 > 120) |
| 157 | 121 | Line too long (126 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_models.py:50"
```

#### `deeplabcut\utils\auxiliaryfunctions_3d.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 206 | 121 | Line too long (121 > 120) |
| 228 | 121 | Line too long (139 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions_3d.py:206"
```

#### `deeplabcut\utils\pseudo_label.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 397 | 121 | Line too long (145 > 120) |
| 399 | 121 | Line too long (133 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\pseudo_label.py:397"
```

#### `tests\pose_estimation_pytorch\other\test_match_predictions_to_gt.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 15 | 121 | Line too long (122 > 120) |
| 78 | 121 | Line too long (125 > 120) |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\other\test_match_predictions_to_gt.py:15"
```

#### `testscript_cli.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 149 | 121 | Line too long (221 > 120) |
| 173 | 121 | Line too long (133 > 120) |

Quick open commands:

```powershell
code -g "testscript_cli.py:149"
```

#### `deeplabcut\__main__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 29 | 121 | Line too long (127 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\__main__.py:29"
```

#### `deeplabcut\gui\tabs\extract_outlier_frames.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 162 | 121 | Line too long (223 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\extract_outlier_frames.py:162"
```

#### `deeplabcut\gui\tabs\modelzoo.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 103 | 121 | Line too long (130 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\modelzoo.py:103"
```

#### `deeplabcut\gui\tabs\refine_tracklets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 231 | 121 | Line too long (223 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\refine_tracklets.py:231"
```

#### `deeplabcut\gui\tabs\train_network.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 97 | 121 | Line too long (121 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\train_network.py:97"
```

#### `deeplabcut\gui\widgets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 529 | 121 | Line too long (223 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\gui\widgets.py:529"
```

#### `deeplabcut\modelzoo\weight_initialization.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 77 | 121 | Line too long (122 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\weight_initialization.py:77"
```

#### `deeplabcut\pose_estimation_pytorch\models\modules\conv_block.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 29 | 121 | Line too long (123 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\modules\conv_block.py:29"
```

#### `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 30 | 121 | Line too long (122 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\necks\transformer.py:30"
```

#### `deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 36 | 121 | Line too long (134 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py:36"
```

#### `deeplabcut\pose_estimation_pytorch\post_processing\match_predictions_to_gt.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 108 | 121 | Line too long (134 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\post_processing\match_predictions_to_gt.py:108"
```

#### `deeplabcut\pose_estimation_tensorflow\core\train.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 151 | 121 | Line too long (139 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\train.py:151"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 413 | 121 | Line too long (124 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:413"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 227 | 121 | Line too long (125 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py:227"
```

#### `deeplabcut\post_processing\analyze_skeleton.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 101 | 121 | Line too long (133 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\analyze_skeleton.py:101"
```

#### `deeplabcut\utils\visualization.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 144 | 121 | Line too long (131 > 120) |

Quick open commands:

```powershell
code -g "deeplabcut\utils\visualization.py:144"
```

#### `examples\JUPYTER\Demo_yourowndata.ipynb` (1)

| Line | Col | Message |
|---:|---:|---|
| 16 | 121 | Line too long (123 > 120) |

Quick open commands:

```powershell
code -g "examples\JUPYTER\Demo_yourowndata.ipynb:16"
```

#### `examples\testscript_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 149 | 121 | Line too long (126 > 120) |

Quick open commands:

```powershell
code -g "examples\testscript_3d.py:149"
```

#### `examples\testscript_deterministicwithResNet152.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 118 | 121 | Line too long (223 > 120) |

Quick open commands:

```powershell
code -g "examples\testscript_deterministicwithResNet152.py:118"
```

#### `examples\testscript_mobilenets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 17 | 121 | Line too long (137 > 120) |

Quick open commands:

```powershell
code -g "examples\testscript_mobilenets.py:17"
```

#### `examples\testscript_openfielddata.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 29 | 121 | Line too long (142 > 120) |

Quick open commands:

```powershell
code -g "examples\testscript_openfielddata.py:29"
```

#### `examples\testscript_pretrained_models.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 32 | 121 | Line too long (144 > 120) |

Quick open commands:

```powershell
code -g "examples\testscript_pretrained_models.py:32"
```

## F401

Count: **331**
Hint: Unused import. Usually safe to delete; verify imports with side effects.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\__init__.py` | 66 |
| `deeplabcut\pose_estimation_pytorch\__init__.py` | 51 |
| `deeplabcut\pose_estimation_pytorch\apis\__init__.py` | 22 |
| `deeplabcut\pose_estimation_pytorch\data\__init__.py` | 19 |
| `deeplabcut\pose_estimation_pytorch\runners\__init__.py` | 18 |
| `deeplabcut\gui\tabs\__init__.py` | 15 |
| `deeplabcut\pose_estimation_pytorch\config\__init__.py` | 12 |
| `deeplabcut\pose_estimation_pytorch\models\modules\__init__.py` | 12 |
| `deeplabcut\pose_estimation_pytorch\models\criterions\__init__.py` | 11 |
| `deeplabcut\pose_estimation_pytorch\models\__init__.py` | 9 |
| `deeplabcut\pose_estimation_pytorch\models\backbones\__init__.py` | 8 |
| `deeplabcut\pose_estimation_pytorch\models\target_generators\__init__.py` | 8 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\__init__.py` | 7 |
| `deeplabcut\pose_estimation_pytorch\models\heads\__init__.py` | 7 |
| `deeplabcut\pose_estimation_pytorch\models\predictors\__init__.py` | 7 |
| `deeplabcut\create_project\__init__.py` | 6 |
| `deeplabcut\pose_estimation_pytorch\modelzoo\__init__.py` | 6 |
| `deeplabcut\core\metrics\__init__.py` | 4 |
| `deeplabcut\pose_estimation_pytorch\models\detectors\__init__.py` | 4 |
| `deeplabcut\pose_tracking_pytorch\processor\__init__.py` | 4 |
| `deeplabcut\generate_training_dataset\__init__.py` | 3 |
| `deeplabcut\modelzoo\generalized_data_converter\__init__.py` | 3 |
| `deeplabcut\pose_estimation_pytorch\models\necks\__init__.py` | 3 |
| `deeplabcut\pose_tracking_pytorch\tracking_utils\__init__.py` | 3 |
| `deeplabcut\gui\window.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\post_processing\__init__.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\__init__.py` | 2 |
| `deeplabcut\pose_tracking_pytorch\__init__.py` | 2 |
| `deeplabcut\pose_tracking_pytorch\model\__init__.py` | 2 |
| `deeplabcut\__main__.py` | 1 |
| `deeplabcut\gui\__init__.py` | 1 |
| `deeplabcut\gui\tabs\create_training_dataset.py` | 1 |
| `deeplabcut\modelzoo\__init__.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\conversion_table\__init__.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\lib\__init__.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\__init__.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\__init__.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\datasets\__init__.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\loss\__init__.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\model\backbones\__init__.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\solver\__init__.py` | 1 |
| `deeplabcut\post_processing\__init__.py` | 1 |

### Details

#### `deeplabcut\__init__.py` (66)

| Line | Col | Message |
|---:|---:|---|
| 16 | 41 | `deeplabcut.version.__version__` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 46 | `deeplabcut.gui.launch_script.launch_dlc` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 23 | 9 | `deeplabcut.gui.tabs.label_frames.label_frames` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 24 | 9 | `deeplabcut.gui.tabs.label_frames.refine_labels` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 26 | 49 | `deeplabcut.gui.tracklet_toolbox.refine_tracklets` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 27 | 40 | `deeplabcut.gui.widgets.SkeletonBuilder` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 31 | 36 | `deeplabcut.core.engine.Engine` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 33 | 5 | `deeplabcut.create_project.add_new_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 34 | 5 | `deeplabcut.create_project.create_new_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 5 | `deeplabcut.create_project.create_new_project_3d` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 36 | 5 | `deeplabcut.create_project.create_pretrained_human_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 37 | 5 | `deeplabcut.create_project.create_pretrained_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 38 | 5 | `deeplabcut.create_project.load_demo_data` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 41 | 5 | `deeplabcut.generate_training_dataset.adddatasetstovideolistandviceversa` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 42 | 5 | `deeplabcut.generate_training_dataset.check_labels` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 43 | 5 | `deeplabcut.generate_training_dataset.comparevideolistsanddatafolders` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 44 | 5 | `deeplabcut.generate_training_dataset.create_multianimaltraining_dataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 45 | 5 | `deeplabcut.generate_training_dataset.create_training_dataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 46 | 5 | `deeplabcut.generate_training_dataset.create_training_dataset_from_existing_split` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 47 | 5 | `deeplabcut.generate_training_dataset.create_training_model_comparison` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 48 | 5 | `deeplabcut.generate_training_dataset.dropannotationfileentriesduetodeletedimages` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 49 | 5 | `deeplabcut.generate_training_dataset.dropduplicatesinannotatinfiles` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 50 | 5 | `deeplabcut.generate_training_dataset.dropimagesduetolackofannotation` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 51 | 5 | `deeplabcut.generate_training_dataset.dropunlabeledframes` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 52 | 5 | `deeplabcut.generate_training_dataset.extract_frames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 53 | 5 | `deeplabcut.generate_training_dataset.mergeandsplit` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 55 | 49 | `deeplabcut.modelzoo.video_inference.video_inference_superanimal` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 57 | 5 | `deeplabcut.utils.analyze_videos_converth5_to_csv` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 58 | 5 | `deeplabcut.utils.analyze_videos_converth5_to_nwb` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 59 | 5 | `deeplabcut.utils.auxfun_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 60 | 5 | `deeplabcut.utils.auxiliaryfunctions` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 61 | 5 | `deeplabcut.utils.convert2_maDLC` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 62 | 5 | `deeplabcut.utils.convertcsv2h5` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 63 | 5 | `deeplabcut.utils.create_labeled_video` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 64 | 5 | `deeplabcut.utils.create_video_with_all_detections` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 65 | 5 | `deeplabcut.utils.plot_trajectories` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 69 | 50 | `deeplabcut.pose_tracking_pytorch.transformer_reID` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 82 | 5 | `deeplabcut.compat.analyze_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 83 | 5 | `deeplabcut.compat.analyze_time_lapse_frames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 84 | 5 | `deeplabcut.compat.analyze_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 85 | 5 | `deeplabcut.compat.convert_detections2tracklets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 86 | 5 | `deeplabcut.compat.create_tracking_dataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 87 | 5 | `deeplabcut.compat.evaluate_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 88 | 5 | `deeplabcut.compat.export_model` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 89 | 5 | `deeplabcut.compat.extract_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 90 | 5 | `deeplabcut.compat.extract_save_all_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 91 | 5 | `deeplabcut.compat.return_evaluate_network_data` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 92 | 5 | `deeplabcut.compat.return_train_network_path` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 93 | 5 | `deeplabcut.compat.train_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 94 | 5 | `deeplabcut.compat.visualize_locrefs` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 95 | 5 | `deeplabcut.compat.visualize_paf` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 96 | 5 | `deeplabcut.compat.visualize_scoremaps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 99 | 5 | `deeplabcut.pose_estimation_3d.calibrate_cameras` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 100 | 5 | `deeplabcut.pose_estimation_3d.check_undistortion` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 101 | 5 | `deeplabcut.pose_estimation_3d.create_labeled_video_3d` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 102 | 5 | `deeplabcut.pose_estimation_3d.triangulate` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 104 | 40 | `deeplabcut.post_processing.analyzeskeleton` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 104 | 57 | `deeplabcut.post_processing.filterpredictions` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 106 | 5 | `deeplabcut.refine_training_dataset.extract_outlier_frames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 107 | 5 | `deeplabcut.refine_training_dataset.find_outliers_in_raw_data` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 108 | 5 | `deeplabcut.refine_training_dataset.merge_datasets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 110 | 55 | `deeplabcut.refine_training_dataset.stitch.stitch_tracklets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 112 | 5 | `deeplabcut.utils.auxfun_videos.CropVideo` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 113 | 5 | `deeplabcut.utils.auxfun_videos.DownSampleVideo` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 114 | 5 | `deeplabcut.utils.auxfun_videos.ShortenVideo` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 115 | 5 | `deeplabcut.utils.auxfun_videos.check_video_integrity` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\__init__.py:16"
```

#### `deeplabcut\pose_estimation_pytorch\__init__.py` (51)

| Line | Col | Message |
|---:|---:|---|
| 11 | 53 | `deeplabcut.pose_estimation_pytorch.config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.apis.VideoIterator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_image_folder` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.pose_estimation_pytorch.apis.build_predictions_dataframe` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 5 | `deeplabcut.pose_estimation_pytorch.apis.convert_detections2tracklets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 5 | `deeplabcut.pose_estimation_pytorch.apis.create_labeled_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 5 | `deeplabcut.pose_estimation_pytorch.apis.create_tracking_dataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluate` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluate_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 23 | 5 | `deeplabcut.pose_estimation_pytorch.apis.extract_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 24 | 5 | `deeplabcut.pose_estimation_pytorch.apis.extract_save_all_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.apis.get_detector_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 26 | 5 | `deeplabcut.pose_estimation_pytorch.apis.get_pose_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.apis.predict` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.apis.superanimal_analyze_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 29 | 5 | `deeplabcut.pose_estimation_pytorch.apis.train` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 30 | 5 | `deeplabcut.pose_estimation_pytorch.apis.train_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 31 | 5 | `deeplabcut.pose_estimation_pytorch.apis.video_inference` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 32 | 5 | `deeplabcut.pose_estimation_pytorch.apis.visualize_predictions` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 5 | `deeplabcut.pose_estimation_pytorch.config.available_detectors` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 36 | 5 | `deeplabcut.pose_estimation_pytorch.config.available_models` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 37 | 5 | `deeplabcut.pose_estimation_pytorch.config.is_model_cond_top_down` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 38 | 5 | `deeplabcut.pose_estimation_pytorch.config.is_model_top_down` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 41 | 5 | `deeplabcut.pose_estimation_pytorch.data.COLLATE_FUNCTIONS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 42 | 5 | `deeplabcut.pose_estimation_pytorch.data.COCOLoader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 43 | 5 | `deeplabcut.pose_estimation_pytorch.data.DLCLoader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 44 | 5 | `deeplabcut.pose_estimation_pytorch.data.GenerativeSampler` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 45 | 5 | `deeplabcut.pose_estimation_pytorch.data.GenSamplingConfig` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 46 | 5 | `deeplabcut.pose_estimation_pytorch.data.Loader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 47 | 5 | `deeplabcut.pose_estimation_pytorch.data.PoseDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 48 | 5 | `deeplabcut.pose_estimation_pytorch.data.PoseDatasetParameters` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 49 | 5 | `deeplabcut.pose_estimation_pytorch.data.Snapshot` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 50 | 5 | `deeplabcut.pose_estimation_pytorch.data.build_transforms` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 51 | 5 | `deeplabcut.pose_estimation_pytorch.data.list_snapshots` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 54 | 5 | `deeplabcut.pose_estimation_pytorch.runners.DetectorInferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 55 | 5 | `deeplabcut.pose_estimation_pytorch.runners.DetectorTrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 56 | 5 | `deeplabcut.pose_estimation_pytorch.runners.DynamicCropper` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 57 | 5 | `deeplabcut.pose_estimation_pytorch.runners.InferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 58 | 5 | `deeplabcut.pose_estimation_pytorch.runners.PoseInferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 59 | 5 | `deeplabcut.pose_estimation_pytorch.runners.PoseTrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 60 | 5 | `deeplabcut.pose_estimation_pytorch.runners.TopDownDynamicCropper` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 61 | 5 | `deeplabcut.pose_estimation_pytorch.runners.TorchSnapshotManager` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 62 | 5 | `deeplabcut.pose_estimation_pytorch.runners.TrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 63 | 5 | `deeplabcut.pose_estimation_pytorch.runners.build_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 64 | 5 | `deeplabcut.pose_estimation_pytorch.runners.build_training_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 65 | 5 | `deeplabcut.pose_estimation_pytorch.runners.get_load_weights_only` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 66 | 5 | `deeplabcut.pose_estimation_pytorch.runners.set_load_weights_only` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 68 | 53 | `deeplabcut.pose_estimation_pytorch.task.Task` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 69 | 54 | `deeplabcut.pose_estimation_pytorch.utils.fix_seeds` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\apis\__init__.py` (22)

| Line | Col | Message |
|---:|---:|---|
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_images.analyze_image_folder` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_images.analyze_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.apis.analyze_images.superanimal_analyze_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluation.evaluate` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluation.evaluate_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluation.predict` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.apis.evaluation.visualize_predictions` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 23 | 60 | `deeplabcut.pose_estimation_pytorch.apis.export.export_model` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.apis.tracking_dataset.create_tracking_dataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.apis.tracklets.convert_detections2tracklets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 31 | 5 | `deeplabcut.pose_estimation_pytorch.apis.training.train` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 32 | 5 | `deeplabcut.pose_estimation_pytorch.apis.training.train_network` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 5 | `deeplabcut.pose_estimation_pytorch.apis.utils.build_predictions_dataframe` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 36 | 5 | `deeplabcut.pose_estimation_pytorch.apis.utils.get_detector_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 37 | 5 | `deeplabcut.pose_estimation_pytorch.apis.utils.get_inference_runners` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 38 | 5 | `deeplabcut.pose_estimation_pytorch.apis.utils.get_pose_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 41 | 5 | `deeplabcut.pose_estimation_pytorch.apis.videos.VideoIterator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 42 | 5 | `deeplabcut.pose_estimation_pytorch.apis.videos.analyze_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 43 | 5 | `deeplabcut.pose_estimation_pytorch.apis.videos.video_inference` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 46 | 5 | `deeplabcut.pose_estimation_pytorch.apis.visualization.create_labeled_images` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 47 | 5 | `deeplabcut.pose_estimation_pytorch.apis.visualization.extract_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 48 | 5 | `deeplabcut.pose_estimation_pytorch.apis.visualization.extract_save_all_maps` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\__init__.py:13"
```

#### `deeplabcut\pose_estimation_pytorch\data\__init__.py` (19)

| Line | Col | Message |
|---:|---:|---|
| 11 | 58 | `deeplabcut.pose_estimation_pytorch.data.base.Loader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 64 | `deeplabcut.pose_estimation_pytorch.data.cocoloader.COCOLoader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 61 | `deeplabcut.pose_estimation_pytorch.data.collate.COLLATE_FUNCTIONS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.data.dataset.PoseDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.data.dataset.PoseDatasetParameters` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 63 | `deeplabcut.pose_estimation_pytorch.data.dlcloader.DLCLoader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 5 | `deeplabcut.pose_estimation_pytorch.data.generative_sampling.GenerativeSampler` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.data.generative_sampling.GenSamplingConfig` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 23 | 59 | `deeplabcut.pose_estimation_pytorch.data.image.top_down_crop` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.data.postprocessor.Postprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 26 | 5 | `deeplabcut.pose_estimation_pytorch.data.postprocessor.build_bottom_up_postprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.data.postprocessor.build_detector_postprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.data.postprocessor.build_top_down_postprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 31 | 5 | `deeplabcut.pose_estimation_pytorch.data.preprocessor.Preprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 32 | 5 | `deeplabcut.pose_estimation_pytorch.data.preprocessor.build_bottom_up_preprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 33 | 5 | `deeplabcut.pose_estimation_pytorch.data.preprocessor.build_top_down_preprocessor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 63 | `deeplabcut.pose_estimation_pytorch.data.snapshots.Snapshot` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 73 | `deeplabcut.pose_estimation_pytorch.data.snapshots.list_snapshots` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 36 | 64 | `deeplabcut.pose_estimation_pytorch.data.transforms.build_transforms` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\runners\__init__.py` (18)

| Line | Col | Message |
|---:|---:|---|
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.runners.base.Runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.runners.base.attempt_snapshot_load` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.runners.base.fix_snapshot_metadata` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.runners.base.get_load_weights_only` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.pose_estimation_pytorch.runners.base.set_load_weights_only` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 60 | `deeplabcut.pose_estimation_pytorch.runners.ctd.CTDTrackingConfig` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.runners.dynamic_cropping.DynamicCropper` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 5 | `deeplabcut.pose_estimation_pytorch.runners.dynamic_cropping.TopDownDynamicCropper` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.runners.inference.DetectorInferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 26 | 5 | `deeplabcut.pose_estimation_pytorch.runners.inference.InferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.runners.inference.PoseInferenceRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.runners.inference.build_inference_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 30 | 63 | `deeplabcut.pose_estimation_pytorch.runners.logger.LOGGER` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 31 | 66 | `deeplabcut.pose_estimation_pytorch.runners.snapshots.TorchSnapshotManager` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 33 | 5 | `deeplabcut.pose_estimation_pytorch.runners.train.DetectorTrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 34 | 5 | `deeplabcut.pose_estimation_pytorch.runners.train.PoseTrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 35 | 5 | `deeplabcut.pose_estimation_pytorch.runners.train.TrainingRunner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 36 | 5 | `deeplabcut.pose_estimation_pytorch.runners.train.build_training_runner` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\__init__.py:13"
```

#### `deeplabcut\gui\tabs\__init__.py` (15)

| Line | Col | Message |
|---:|---:|---|
| 11 | 48 | `deeplabcut.gui.tabs.analyze_videos.AnalyzeVideos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 48 | `deeplabcut.gui.tabs.create_project.ProjectCreator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 57 | `deeplabcut.gui.tabs.create_training_dataset.CreateTrainingDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 47 | `deeplabcut.gui.tabs.create_videos.CreateVideos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 50 | `deeplabcut.gui.tabs.evaluate_network.EvaluateNetwork` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 48 | `deeplabcut.gui.tabs.extract_frames.ExtractFrames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 56 | `deeplabcut.gui.tabs.extract_outlier_frames.ExtractOutlierFrames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 46 | `deeplabcut.gui.tabs.label_frames.LabelFrames` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 48 | `deeplabcut.gui.tabs.manage_project.ManageProject` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 42 | `deeplabcut.gui.tabs.modelzoo.ModelZoo` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 46 | `deeplabcut.gui.tabs.open_project.OpenProject` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 50 | `deeplabcut.gui.tabs.refine_tracklets.RefineTracklets` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 23 | 47 | `deeplabcut.gui.tabs.train_network.TrainNetwork` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 24 | 58 | `deeplabcut.gui.tabs.unsupervised_id_tracking.UnsupervizedIdTracking` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 46 | `deeplabcut.gui.tabs.video_editor.VideoEditor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\config\__init__.py` (12)

| Line | Col | Message |
|---:|---:|---|
| 13 | 5 | `deeplabcut.core.config.pretty_print` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.core.config.read_config_as_dict` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.core.config.write_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 5 | `deeplabcut.pose_estimation_pytorch.config.make_pose_config.make_basic_project_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 5 | `deeplabcut.pose_estimation_pytorch.config.make_pose_config.make_pytorch_pose_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 5 | `deeplabcut.pose_estimation_pytorch.config.make_pose_config.make_pytorch_test_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 23 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.available_detectors` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 24 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.available_models` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.is_model_cond_top_down` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 26 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.is_model_top_down` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.update_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.config.utils.update_config_by_dotpath` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\config\__init__.py:13"
```

#### `deeplabcut\pose_estimation_pytorch\models\modules\__init__.py` (12)

| Line | Col | Message |
|---:|---:|---|
| 11 | 75 | `deeplabcut.pose_estimation_pytorch.models.modules.coam_module.CoAMBlock` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 86 | `deeplabcut.pose_estimation_pytorch.models.modules.coam_module.SelfAttentionModule_CoAM` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.conv_block.AdaptBlock` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.conv_block.BasicBlock` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.conv_block.Bottleneck` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.conv_module.HighResolutionModule` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.gated_attention_unit.GatedAttentionUnit` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 24 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.kpt_encoders.KEYPOINT_ENCODERS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.kpt_encoders.BaseKeypointEncoder` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 26 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.kpt_encoders.ColoredKeypointEncoder` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.kpt_encoders.StackedKeypointEncoder` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 30 | 5 | `deeplabcut.pose_estimation_pytorch.models.modules.norm.ScaleNorm` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\modules\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\criterions\__init__.py` (11)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.aggregators.WeightedLossAggregator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.base.CRITERIONS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.base.LOSS_AGGREGATORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.base.BaseCriterion` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.base.BaseLossAggregator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.dekr.DEKRHeatmapLoss` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.dekr.DEKROffsetLoss` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.kl_discrete.KLDiscreteLoss` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.weighted.WeightedBCECriterion` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 29 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.weighted.WeightedHuberCriterion` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 30 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.weighted.WeightedMSECriterion` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\criterions\__init__.py:12"
```

#### `deeplabcut\pose_estimation_pytorch\models\__init__.py` (9)

| Line | Col | Message |
|---:|---:|---|
| 11 | 70 | `deeplabcut.pose_estimation_pytorch.models.backbones.base.BACKBONES` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.CRITERIONS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.models.criterions.LOSS_AGGREGATORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 65 | `deeplabcut.pose_estimation_pytorch.models.detectors.DETECTORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 66 | `deeplabcut.pose_estimation_pytorch.models.heads.base.HEADS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 61 | `deeplabcut.pose_estimation_pytorch.models.model.PoseModel` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 66 | `deeplabcut.pose_estimation_pytorch.models.necks.base.NECKS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 66 | `deeplabcut.pose_estimation_pytorch.models.predictors.PREDICTORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.TARGET_GENERATORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\backbones\__init__.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.models.backbones.base.BACKBONES` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.backbones.base.BaseBackbone` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 77 | `deeplabcut.pose_estimation_pytorch.models.backbones.cond_prenet.CondPreNet` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 73 | `deeplabcut.pose_estimation_pytorch.models.backbones.cspnext.CSPNeXt` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 71 | `deeplabcut.pose_estimation_pytorch.models.backbones.hrnet.HRNet` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 76 | `deeplabcut.pose_estimation_pytorch.models.backbones.hrnet_coam.HRNetCoAM` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 72 | `deeplabcut.pose_estimation_pytorch.models.backbones.resnet.DLCRNet` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 81 | `deeplabcut.pose_estimation_pytorch.models.backbones.resnet.ResNet` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\backbones\__init__.py:12"
```

#### `deeplabcut\pose_estimation_pytorch\models\target_generators\__init__.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.base.TARGET_GENERATORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.base.BaseGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.base.SequentialGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.dekr_targets.DEKRGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 20 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.heatmap_targets.HeatmapGaussianGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 21 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.heatmap_targets.HeatmapPlateauGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 24 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.pafs_targets.PartAffinityFieldGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 27 | 5 | `deeplabcut.pose_estimation_pytorch.models.target_generators.sim_cc.SimCCGenerator` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\target_generators\__init__.py:12"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\__init__.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 11 | 19 | `.coco.COCOPoseDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 21 | `.ma_dlc.MaDLCPoseDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 31 | `.ma_dlc_dataframe.MaDLCDataFrame` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 26 | `.materialize.mat_func_factory` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 20 | `.multi.MultiSourceDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 25 | `.single_dlc.SingleDLCPoseDataset` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 35 | `.single_dlc_dataframe.SingleDLCDataFrame` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\heads\__init__.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 11 | 66 | `deeplabcut.pose_estimation_pytorch.models.heads.base.HEADS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 73 | `deeplabcut.pose_estimation_pytorch.models.heads.base.BaseHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 66 | `deeplabcut.pose_estimation_pytorch.models.heads.dekr.DEKRHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 69 | `deeplabcut.pose_estimation_pytorch.models.heads.dlcrnet.DLCRNetHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 72 | `deeplabcut.pose_estimation_pytorch.models.heads.rtmcc_head.RTMCCHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 73 | `deeplabcut.pose_estimation_pytorch.models.heads.simple_head.HeatmapHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 73 | `deeplabcut.pose_estimation_pytorch.models.heads.transformer.TransformerHead` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\heads\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\predictors\__init__.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.base.PREDICTORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.base.BasePredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.dekr_predictor.DEKRPredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 19 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.identity_predictor.IdentityPredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 22 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.paf_predictor.PartAffinityFieldPredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 25 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.sim_cc.SimCCPredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 5 | `deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor.HeatmapPredictor` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\predictors\__init__.py:12"
```

#### `deeplabcut\create_project\__init__.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 11 | 43 | `deeplabcut.create_project.add.add_new_videos` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 49 | `deeplabcut.create_project.demo_data.load_demo_data` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.create_project.modelzoo.create_pretrained_human_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.create_project.modelzoo.create_pretrained_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 43 | `deeplabcut.create_project.new.create_new_project` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 18 | 46 | `deeplabcut.create_project.new_3d.create_new_project_3d` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\modelzoo\__init__.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.download_super_animal_snapshot` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.get_snapshot_folder_path` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.get_super_animal_model_config_path` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.get_super_animal_project_config_path` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.get_super_animal_snapshot_path` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.pose_estimation_pytorch.modelzoo.utils.load_super_animal_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\modelzoo\__init__.py:12"
```

#### `deeplabcut\core\metrics\__init__.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 11 | 18 | `.api.compute_metrics` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 35 | `.api.prepare_evaluation_data` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 19 | `.bbox.compute_bbox_metrics` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 23 | `.identity.compute_identity_scores` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\core\metrics\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\detectors\__init__.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.models.detectors.base.DETECTORS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.models.detectors.base.BaseDetector` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 76 | `deeplabcut.pose_estimation_pytorch.models.detectors.fasterRCNN.FasterRCNN` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 69 | `deeplabcut.pose_estimation_pytorch.models.detectors.ssd.SSDLite` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\detectors\__init__.py:12"
```

#### `deeplabcut\pose_tracking_pytorch\processor\__init__.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 13 | 5 | `.processor.default_device` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `.processor.do_dlc_inference` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 5 | `.processor.do_dlc_pair_inference` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `.processor.do_dlc_train` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\processor\__init__.py:13"
```

#### `deeplabcut\generate_training_dataset\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 15 | 5 | `deeplabcut.generate_training_dataset.metadata.DataSplit` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 16 | 5 | `deeplabcut.generate_training_dataset.metadata.ShuffleMetadata` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 17 | 5 | `deeplabcut.generate_training_dataset.metadata.TrainingDatasetMetadata` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\__init__.py:15"
```

#### `deeplabcut\modelzoo\generalized_data_converter\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 11 | 20 | `.utils.add_skeleton` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 34 | `.utils.create_modelprefix` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 54 | `.utils.customized_colormap` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\__init__.py:11"
```

#### `deeplabcut\pose_estimation_pytorch\models\necks\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 11 | 66 | `deeplabcut.pose_estimation_pytorch.models.necks.base.NECKS` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 11 | 73 | `deeplabcut.pose_estimation_pytorch.models.necks.base.BaseNeck` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 73 | `deeplabcut.pose_estimation_pytorch.models.necks.transformer.Transformer` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\necks\__init__.py:11"
```

#### `deeplabcut\pose_tracking_pytorch\tracking_utils\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `.preprocessing.convert_coord_from_img_space_to_feature_space` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `.preprocessing.load_features_from_coord` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 14 | 5 | `.preprocessing.query_feature_by_coord_in_img_space` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\tracking_utils\__init__.py:12"
```

#### `deeplabcut\gui\window.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 196 | 24 | `tensorflow` imported but unused; consider using `importlib.util.find_spec` to test for availability |
| 696 | 62 | `deeplabcut.pose_tracking_pytorch.transformer_reID` imported but unused; consider using `importlib.util.find_spec` to test for availability |

Quick open commands:

```powershell
code -g "deeplabcut\gui\window.py:196"
```

#### `deeplabcut\pose_estimation_pytorch\post_processing\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 12 | 5 | `deeplabcut.pose_estimation_pytorch.post_processing.match_predictions_to_gt.oks_match_prediction_to_gt` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 13 | 5 | `deeplabcut.pose_estimation_pytorch.post_processing.match_predictions_to_gt.rmse_match_prediction_to_gt` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\post_processing\__init__.py:12"
```

#### `deeplabcut\pose_estimation_tensorflow\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 18 | 15 | `._tf_legacy` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 28 | 58 | `deeplabcut.pose_estimation_tensorflow.export.export_model` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\__init__.py:18"
```

#### `deeplabcut\pose_tracking_pytorch\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 12 | 19 | `.apis.transformer_reID` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 15 | 33 | `.train_dlctransreid.train_tracking_transformer` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\__init__.py:12"
```

#### `deeplabcut\pose_tracking_pytorch\model\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 12 | 25 | `.make_model.build_dlc_transformer` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |
| 12 | 48 | `.make_model.make_dlc_model` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\model\__init__.py:12"
```

#### `deeplabcut\__main__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 15 | 16 | `PySide6` imported but unused; consider using `importlib.util.find_spec` to test for availability |

Quick open commands:

```powershell
code -g "deeplabcut\__main__.py:15"
```

#### `deeplabcut\gui\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 15 | 8 | `qtpy` imported but unused |

Quick open commands:

```powershell
code -g "deeplabcut\gui\__init__.py:15"
```

#### `deeplabcut\gui\tabs\create_training_dataset.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 248 | 28 | `tensorflow` imported but unused; consider using `importlib.util.find_spec` to test for availability |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\create_training_dataset.py:248"
```

#### `deeplabcut\modelzoo\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 55 | `deeplabcut.modelzoo.weight_initialization.build_weight_init` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\__init__.py:11"
```

#### `deeplabcut\modelzoo\generalized_data_converter\conversion_table\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 31 | `.conversion_table.get_conversion_table` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\conversion_table\__init__.py:11"
```

#### `deeplabcut\pose_estimation_tensorflow\lib\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 15 | 8 | `deeplabcut.core.trackingutils` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\lib\__init__.py:15"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 18 | `.api.SpatiotemporalAdaptation` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\__init__.py:11"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 35 | `.spatiotemporal_adapt.SpatiotemporalAdaptation` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\__init__.py:11"
```

#### `deeplabcut\pose_tracking_pytorch\datasets\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 12 | 30 | `.make_dataloader.make_dlc_dataloader` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\datasets\__init__.py:12"
```

#### `deeplabcut\pose_tracking_pytorch\loss\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 24 | `.make_loss.easy_triplet_loss` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\loss\__init__.py:11"
```

#### `deeplabcut\pose_tracking_pytorch\model\backbones\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 26 | `.vit_pytorch.dlc_base_kpt_TransReID` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\model\backbones\__init__.py:11"
```

#### `deeplabcut\pose_tracking_pytorch\solver\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 11 | 29 | `.make_optimizer.make_easy_optimizer` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\solver\__init__.py:11"
```

#### `deeplabcut\post_processing\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 21 | 57 | `deeplabcut.post_processing.analyze_skeleton.analyzeskeleton` imported but unused; consider removing, adding to `__all__`, or using a redundant alias |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\__init__.py:21"
```

## B905

Count: **176**

### Files affected

| File | Count |
|---|---:|
| `docs\recipes\flip_and_rotate.ipynb` | 18 |
| `deeplabcut\refine_training_dataset\stitch.py` | 12 |
| `deeplabcut\core\inferenceutils.py` | 8 |
| `deeplabcut\refine_training_dataset\tracklets.py` | 8 |
| `deeplabcut\utils\visualization.py` | 8 |
| `deeplabcut\core\crossvalutils.py` | 7 |
| `deeplabcut\utils\pseudo_label.py` | 6 |
| `deeplabcut\pose_estimation_pytorch\apis\prune_paf_graph.py` | 5 |
| `deeplabcut\utils\make_labeled_video.py` | 5 |
| `examples\COLAB\COLAB_HumanPose_with_RTMPose.ipynb` | 5 |
| `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` | 4 |
| `deeplabcut\pose_estimation_pytorch\data\preprocessor.py` | 3 |
| `tests\pose_estimation_pytorch\data\test_transforms.py` | 3 |
| `tests\pose_estimation_pytorch\runners\test_runners_inference.py` | 3 |
| `deeplabcut\core\metrics\distance_metrics.py` | 2 |
| `deeplabcut\core\trackingutils.py` | 2 |
| `deeplabcut\create_project\add.py` | 2 |
| `deeplabcut\create_project\new.py` | 2 |
| `deeplabcut\modelzoo\webapp\inference.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\apis\analyze_images.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\apis\evaluation.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\apis\visualization.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\data\postprocessor.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\data\transforms.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\runners\logger.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\runners\train.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 2 |
| `tests\test_pose_multianimal_imgaug.py` | 2 |
| `tests\test_predict_supermodel.py` | 2 |
| `deeplabcut\benchmark\metrics.py` | 1 |
| `deeplabcut\core\metrics\bbox.py` | 1 |
| `deeplabcut\core\metrics\identity.py` | 1 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 1 |
| `deeplabcut\generate_training_dataset\metadata.py` | 1 |
| `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` | 1 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 1 |
| `deeplabcut\gui\tabs\create_videos.py` | 1 |
| `deeplabcut\gui\tabs\evaluate_network.py` | 1 |
| `deeplabcut\gui\tracklet_toolbox.py` | 1 |
| `deeplabcut\gui\widgets.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\utils.py` | 1 |
| `deeplabcut\modelzoo\utils.py` | 1 |
| `deeplabcut\pose_estimation_3d\plotting3D.py` | 1 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\apis\tracklets.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\apis\utils.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\data\utils.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\backbones\hrnet_coam.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\heads\dlcrnet.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\modelzoo\utils.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\post_processing\identity.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\post_processing\nms.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\schedulers.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\predict_multianimal.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\export.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\solver\scheduler.py` | 1 |
| `deeplabcut\post_processing\analyze_skeleton.py` | 1 |
| `deeplabcut\post_processing\filtering.py` | 1 |
| `deeplabcut\utils\auxfun_videos.py` | 1 |
| `deeplabcut\utils\auxiliaryfunctions_3d.py` | 1 |
| `deeplabcut\utils\skeleton.py` | 1 |
| `examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb` | 1 |
| `examples\testscript_multianimal.py` | 1 |
| `examples\testscript_transreid.py` | 1 |
| `examples\utils.py` | 1 |
| `tests\generate_training_dataset\test_trainset_metadata.py` | 1 |
| `tests\pose_estimation_pytorch\data\test_data_ctd.py` | 1 |
| `tests\pose_estimation_pytorch\data\test_postprocessor.py` | 1 |
| `tests\pose_estimation_pytorch\data\test_preprocessor.py` | 1 |
| `tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py` | 1 |
| `tests\test_inferenceutils.py` | 1 |
| `tests\test_stitcher.py` | 1 |

### Details

#### `docs\recipes\flip_and_rotate.ipynb` (18)

| Line | Col | Message |
|---:|---:|---|
| 8 | 34 | `zip()` without an explicit `strict=` parameter |
| 14 | 34 | `zip()` without an explicit `strict=` parameter |
| 15 | 34 | `zip()` without an explicit `strict=` parameter |
| 15 | 34 | `zip()` without an explicit `strict=` parameter |
| 15 | 34 | `zip()` without an explicit `strict=` parameter |
| 15 | 34 | `zip()` without an explicit `strict=` parameter |
| 18 | 34 | `zip()` without an explicit `strict=` parameter |
| 26 | 34 | `zip()` without an explicit `strict=` parameter |
| 27 | 34 | `zip()` without an explicit `strict=` parameter |
| 35 | 36 | `zip()` without an explicit `strict=` parameter |
| 35 | 36 | `zip()` without an explicit `strict=` parameter |
| 35 | 41 | `zip()` without an explicit `strict=` parameter |
| 35 | 41 | `zip()` without an explicit `strict=` parameter |
| 35 | 41 | `zip()` without an explicit `strict=` parameter |
| 37 | 36 | `zip()` without an explicit `strict=` parameter |
| 58 | 49 | `zip()` without an explicit `strict=` parameter |
| 58 | 49 | `zip()` without an explicit `strict=` parameter |
| 58 | 49 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "docs\recipes\flip_and_rotate.ipynb:35"
```

#### `deeplabcut\refine_training_dataset\stitch.py` (12)

| Line | Col | Message |
|---:|---:|---|
| 62 | 58 | `zip()` without an explicit `strict=` parameter |
| 526 | 30 | `zip()` without an explicit `strict=` parameter |
| 564 | 56 | `zip()` without an explicit `strict=` parameter |
| 627 | 31 | `zip()` without an explicit `strict=` parameter |
| 630 | 31 | `zip()` without an explicit `strict=` parameter |
| 631 | 31 | `zip()` without an explicit `strict=` parameter |
| 632 | 31 | `zip()` without an explicit `strict=` parameter |
| 678 | 38 | `zip()` without an explicit `strict=` parameter |
| 679 | 38 | `zip()` without an explicit `strict=` parameter |
| 915 | 36 | `zip()` without an explicit `strict=` parameter |
| 928 | 29 | `zip()` without an explicit `strict=` parameter |
| 950 | 30 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\stitch.py:62"
```

#### `deeplabcut\core\inferenceutils.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 157 | 21 | `zip()` without an explicit `strict=` parameter |
| 423 | 49 | `zip()` without an explicit `strict=` parameter |
| 426 | 29 | `zip()` without an explicit `strict=` parameter |
| 460 | 21 | `zip()` without an explicit `strict=` parameter |
| 484 | 33 | `zip()` without an explicit `strict=` parameter |
| 757 | 24 | `zip()` without an explicit `strict=` parameter |
| 1033 | 25 | `zip()` without an explicit `strict=` parameter |
| 1084 | 24 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\inferenceutils.py:157"
```

#### `deeplabcut\refine_training_dataset\tracklets.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 109 | 30 | `zip()` without an explicit `strict=` parameter |
| 163 | 41 | `zip()` without an explicit `strict=` parameter |
| 199 | 25 | `zip()` without an explicit `strict=` parameter |
| 257 | 49 | `zip()` without an explicit `strict=` parameter |
| 259 | 52 | `zip()` without an explicit `strict=` parameter |
| 305 | 25 | `zip()` without an explicit `strict=` parameter |
| 318 | 21 | `zip()` without an explicit `strict=` parameter |
| 328 | 21 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\tracklets.py:109"
```

#### `deeplabcut\utils\visualization.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 163 | 48 | `zip()` without an explicit `strict=` parameter |
| 183 | 30 | `zip()` without an explicit `strict=` parameter |
| 337 | 35 | `zip()` without an explicit `strict=` parameter |
| 346 | 41 | `zip()` without an explicit `strict=` parameter |
| 363 | 26 | `zip()` without an explicit `strict=` parameter |
| 364 | 23 | `zip()` without an explicit `strict=` parameter |
| 399 | 30 | `zip()` without an explicit `strict=` parameter |
| 420 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\visualization.py:163"
```

#### `deeplabcut\core\crossvalutils.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 146 | 47 | `zip()` without an explicit `strict=` parameter |
| 152 | 34 | `zip()` without an explicit `strict=` parameter |
| 225 | 17 | `zip()` without an explicit `strict=` parameter |
| 343 | 40 | `zip()` without an explicit `strict=` parameter |
| 345 | 17 | `zip()` without an explicit `strict=` parameter |
| 349 | 24 | `zip()` without an explicit `strict=` parameter |
| 371 | 27 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\crossvalutils.py:146"
```

#### `deeplabcut\utils\pseudo_label.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 265 | 24 | `zip()` without an explicit `strict=` parameter |
| 276 | 35 | `zip()` without an explicit `strict=` parameter |
| 307 | 32 | `zip()` without an explicit `strict=` parameter |
| 320 | 24 | `zip()` without an explicit `strict=` parameter |
| 420 | 57 | `zip()` without an explicit `strict=` parameter |
| 434 | 39 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\pseudo_label.py:265"
```

#### `deeplabcut\pose_estimation_pytorch\apis\prune_paf_graph.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 216 | 52 | `zip()` without an explicit `strict=` parameter |
| 222 | 34 | `zip()` without an explicit `strict=` parameter |
| 259 | 17 | `zip()` without an explicit `strict=` parameter |
| 263 | 24 | `zip()` without an explicit `strict=` parameter |
| 281 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\prune_paf_graph.py:216"
```

#### `deeplabcut\utils\make_labeled_video.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 62 | 17 | `zip()` without an explicit `strict=` parameter |
| 1092 | 35 | `zip()` without an explicit `strict=` parameter |
| 1099 | 41 | `zip()` without an explicit `strict=` parameter |
| 1116 | 25 | `zip()` without an explicit `strict=` parameter |
| 1134 | 37 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\make_labeled_video.py:62"
```

#### `examples\COLAB\COLAB_HumanPose_with_RTMPose.ipynb` (5)

| Line | Col | Message |
|---:|---:|---|
| 8 | 38 | `zip()` without an explicit `strict=` parameter |
| 35 | 49 | `zip()` without an explicit `strict=` parameter |
| 39 | 49 | `zip()` without an explicit `strict=` parameter |
| 61 | 37 | `zip()` without an explicit `strict=` parameter |
| 69 | 77 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "examples\COLAB\COLAB_HumanPose_with_RTMPose.ipynb:35"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 292 | 49 | `zip()` without an explicit `strict=` parameter |
| 318 | 49 | `zip()` without an explicit `strict=` parameter |
| 411 | 34 | `zip()` without an explicit `strict=` parameter |
| 428 | 34 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_multianimal.py:292"
```

#### `deeplabcut\pose_estimation_pytorch\data\preprocessor.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 273 | 54 | `zip()` without an explicit `strict=` parameter |
| 276 | 89 | `zip()` without an explicit `strict=` parameter |
| 280 | 97 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\preprocessor.py:273"
```

#### `tests\pose_estimation_pytorch\data\test_transforms.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 59 | 32 | `zip()` without an explicit `strict=` parameter |
| 224 | 36 | `zip()` without an explicit `strict=` parameter |
| 272 | 30 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_transforms.py:59"
```

#### `tests\pose_estimation_pytorch\runners\test_runners_inference.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 91 | 17 | `zip()` without an explicit `strict=` parameter |
| 143 | 17 | `zip()` without an explicit `strict=` parameter |
| 145 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\test_runners_inference.py:91"
```

#### `deeplabcut\core\metrics\distance_metrics.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 364 | 56 | `zip()` without an explicit `strict=` parameter |
| 402 | 56 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\metrics\distance_metrics.py:364"
```

#### `deeplabcut\core\trackingutils.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 447 | 29 | `zip()` without an explicit `strict=` parameter |
| 706 | 25 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\trackingutils.py:447"
```

#### `deeplabcut\create_project\add.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 77 | 25 | `zip()` without an explicit `strict=` parameter |
| 87 | 25 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\add.py:77"
```

#### `deeplabcut\create_project\new.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 185 | 25 | `zip()` without an explicit `strict=` parameter |
| 190 | 25 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new.py:185"
```

#### `deeplabcut\modelzoo\webapp\inference.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 103 | 29 | `zip()` without an explicit `strict=` parameter |
| 106 | 71 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\webapp\inference.py:103"
```

#### `deeplabcut\pose_estimation_pytorch\apis\analyze_images.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 532 | 28 | `zip()` without an explicit `strict=` parameter |
| 541 | 80 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\analyze_images.py:532"
```

#### `deeplabcut\pose_estimation_pytorch\apis\evaluation.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 94 | 36 | `zip()` without an explicit `strict=` parameter |
| 97 | 80 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\evaluation.py:94"
```

#### `deeplabcut\pose_estimation_pytorch\apis\visualization.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 149 | 49 | `zip()` without an explicit `strict=` parameter |
| 366 | 43 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\visualization.py:149"
```

#### `deeplabcut\pose_estimation_pytorch\data\postprocessor.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 395 | 54 | `zip()` without an explicit `strict=` parameter |
| 518 | 46 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\postprocessor.py:395"
```

#### `deeplabcut\pose_estimation_pytorch\data\transforms.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 498 | 31 | `zip()` without an explicit `strict=` parameter |
| 641 | 59 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\transforms.py:498"
```

#### `deeplabcut\pose_estimation_pytorch\runners\logger.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 241 | 48 | `zip()` without an explicit `strict=` parameter |
| 505 | 35 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\logger.py:241"
```

#### `deeplabcut\pose_estimation_pytorch\runners\train.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 506 | 40 | `zip()` without an explicit `strict=` parameter |
| 638 | 72 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\train.py:506"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 90 | 44 | `zip()` without an explicit `strict=` parameter |
| 411 | 49 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:90"
```

#### `tests\test_pose_multianimal_imgaug.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 73 | 49 | `zip()` without an explicit `strict=` parameter |
| 78 | 32 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\test_pose_multianimal_imgaug.py:73"
```

#### `tests\test_predict_supermodel.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 26 | 46 | `zip()` without an explicit `strict=` parameter |
| 48 | 63 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\test_predict_supermodel.py:26"
```

#### `deeplabcut\benchmark\metrics.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 58 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\benchmark\metrics.py:58"
```

#### `deeplabcut\core\metrics\bbox.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 100 | 28 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\metrics\bbox.py:100"
```

#### `deeplabcut\core\metrics\identity.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 67 | 53 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\core\metrics\identity.py:67"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 498 | 32 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:498"
```

#### `deeplabcut\generate_training_dataset\metadata.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 140 | 45 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\metadata.py:140"
```

#### `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 387 | 59 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py:387"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1097 | 63 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:1097"
```

#### `deeplabcut\gui\tabs\create_videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 273 | 58 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\create_videos.py:273"
```

#### `deeplabcut\gui\tabs\evaluate_network.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 50 | 37 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\evaluate_network.py:50"
```

#### `deeplabcut\gui\tracklet_toolbox.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 772 | 46 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tracklet_toolbox.py:772"
```

#### `deeplabcut\gui\widgets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 663 | 21 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\gui\widgets.py:663"
```

#### `deeplabcut\modelzoo\generalized_data_converter\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 187 | 28 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\utils.py:187"
```

#### `deeplabcut\modelzoo\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 184 | 17 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\utils.py:184"
```

#### `deeplabcut\pose_estimation_3d\plotting3D.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 265 | 31 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\plotting3D.py:265"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 474 | 38 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:474"
```

#### `deeplabcut\pose_estimation_pytorch\apis\tracklets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 276 | 51 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\tracklets.py:276"
```

#### `deeplabcut\pose_estimation_pytorch\apis\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 465 | 17 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\utils.py:465"
```

#### `deeplabcut\pose_estimation_pytorch\data\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 495 | 26 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\utils.py:495"
```

#### `deeplabcut\pose_estimation_pytorch\models\backbones\hrnet_coam.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 189 | 43 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\backbones\hrnet_coam.py:189"
```

#### `deeplabcut\pose_estimation_pytorch\models\heads\dlcrnet.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 120 | 63 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\heads\dlcrnet.py:120"
```

#### `deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 196 | 35 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py:196"
```

#### `deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 388 | 23 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\predictors\paf_predictor.py:388"
```

#### `deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 107 | 30 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py:107"
```

#### `deeplabcut\pose_estimation_pytorch\modelzoo\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 160 | 27 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\modelzoo\utils.py:160"
```

#### `deeplabcut\pose_estimation_pytorch\post_processing\identity.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 42 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\post_processing\identity.py:42"
```

#### `deeplabcut\pose_estimation_pytorch\post_processing\nms.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 91 | 39 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\post_processing\nms.py:91"
```

#### `deeplabcut\pose_estimation_pytorch\runners\schedulers.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 129 | 29 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\schedulers.py:129"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 331 | 42 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py:331"
```

#### `deeplabcut\pose_estimation_tensorflow\core\predict_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 90 | 26 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\predict_multianimal.py:90"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 391 | 32 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:391"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 472 | 32 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:472"
```

#### `deeplabcut\pose_estimation_tensorflow\export.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 325 | 21 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\export.py:325"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 368 | 50 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py:368"
```

#### `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 256 | 22 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\visualizemaps.py:256"
```

#### `deeplabcut\pose_tracking_pytorch\solver\scheduler.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 96 | 35 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\solver\scheduler.py:96"
```

#### `deeplabcut\post_processing\analyze_skeleton.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 60 | 54 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\analyze_skeleton.py:60"
```

#### `deeplabcut\post_processing\filtering.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 57 | 39 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\filtering.py:57"
```

#### `deeplabcut\utils\auxfun_videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 274 | 42 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_videos.py:274"
```

#### `deeplabcut\utils\auxiliaryfunctions_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 317 | 19 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions_3d.py:317"
```

#### `deeplabcut\utils\skeleton.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 171 | 21 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "deeplabcut\utils\skeleton.py:171"
```

#### `examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb` (1)

| Line | Col | Message |
|---:|---:|---|
| 22 | 37 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb:22"
```

#### `examples\testscript_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 85 | 17 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "examples\testscript_multianimal.py:85"
```

#### `examples\testscript_transreid.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 81 | 17 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "examples\testscript_transreid.py:81"
```

#### `examples\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 102 | 34 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "examples\utils.py:102"
```

#### `tests\generate_training_dataset\test_trainset_metadata.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 326 | 34 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\generate_training_dataset\test_trainset_metadata.py:326"
```

#### `tests\pose_estimation_pytorch\data\test_data_ctd.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 151 | 91 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_data_ctd.py:151"
```

#### `tests\pose_estimation_pytorch\data\test_postprocessor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 304 | 28 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_postprocessor.py:304"
```

#### `tests\pose_estimation_pytorch\data\test_preprocessor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 156 | 49 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_preprocessor.py:156"
```

#### `tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 166 | 47 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py:166"
```

#### `tests\test_inferenceutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 30 | 17 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\test_inferenceutils.py:30"
```

#### `tests\test_stitcher.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 103 | 10 | `zip()` without an explicit `strict=` parameter |

Quick open commands:

```powershell
code -g "tests\test_stitcher.py:103"
```

## F841

Count: **141**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 74 |
| `deeplabcut\utils\pseudo_label.py` | 7 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 5 |
| `deeplabcut\modelzoo\generalized_data_converter\utils.py` | 4 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 3 |
| `deeplabcut\pose_estimation_pytorch\models\predictors\dekr_predictor.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\export.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 3 |
| `deeplabcut\pose_estimation_pytorch\models\necks\layers.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` | 2 |
| `tests\pose_estimation_pytorch\runners\bottum_up.py` | 2 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 1 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 1 |
| `deeplabcut\gui\tabs\modelzoo.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` | 1 |
| `deeplabcut\modelzoo\utils.py` | 1 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 1 |
| `deeplabcut\pose_estimation_3d\plotting3D.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\apis\videos.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\nnets\multi.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\training.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\processor\processor.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\train_dlctransreid.py` | 1 |
| `deeplabcut\refine_training_dataset\outlier_frames.py` | 1 |
| `deeplabcut\utils\auxfun_videos.py` | 1 |
| `examples\testscript_mobilenets.py` | 1 |
| `tests\pose_estimation_pytorch\apis\test_apis_evaluate.py` | 1 |
| `tests\pose_estimation_pytorch\config\test_make_pose_config.py` | 1 |
| `tests\pose_estimation_pytorch\data\test_transforms.py` | 1 |
| `tests\pose_estimation_pytorch\modelzoo\test_load_superanimal_models.py` | 1 |
| `tests\pose_estimation_pytorch\other\test_api_utils.py` | 1 |
| `tests\pose_estimation_pytorch\runners\test_runners_inference.py` | 1 |
| `tests\test_auxiliaryfunctions.py` | 1 |
| `tests\test_pose_multianimal_imgaug.py` | 1 |

### Details

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (74)

| Line | Col | Message |
|---:|---:|---|
| 80 | 9 | Local variable `Task` is assigned to but never used |
| 81 | 9 | Local variable `project_path` is assigned to but never used |
| 82 | 9 | Local variable `scorer` is assigned to but never used |
| 83 | 9 | Local variable `date` is assigned to but never used |
| 84 | 9 | Local variable `video_sets` is assigned to but never used |
| 85 | 9 | Local variable `skeleton` is assigned to but never used |
| 86 | 9 | Local variable `bodyparts` is assigned to but never used |
| 87 | 9 | Local variable `start` is assigned to but never used |
| 88 | 9 | Local variable `stop` is assigned to but never used |
| 89 | 9 | Local variable `numframes2pick` is assigned to but never used |
| 90 | 9 | Local variable `skeleton_color` is assigned to but never used |
| 91 | 9 | Local variable `pcutoff` is assigned to but never used |
| 92 | 9 | Local variable `dotsize` is assigned to but never used |
| 93 | 9 | Local variable `alphavalue` is assigned to but never used |
| 94 | 9 | Local variable `colormap` is assigned to but never used |
| 95 | 9 | Local variable `TrainingFraction` is assigned to but never used |
| 96 | 9 | Local variable `iteration` is assigned to but never used |
| 97 | 9 | Local variable `default_net_type` is assigned to but never used |
| 98 | 9 | Local variable `default_augmenter` is assigned to but never used |
| 99 | 9 | Local variable `snapshotindex` is assigned to but never used |
| 100 | 9 | Local variable `batch_size` is assigned to but never used |
| 101 | 9 | Local variable `cropping` is assigned to but never used |
| 102 | 9 | Local variable `croppedtraining` is assigned to but never used |
| 103 | 9 | Local variable `multianimalproject` is assigned to but never used |
| 104 | 9 | Local variable `uniquebodyparts` is assigned to but never used |
| 105 | 9 | Local variable `x1` is assigned to but never used |
| 106 | 9 | Local variable `x2` is assigned to but never used |
| 107 | 9 | Local variable `y1` is assigned to but never used |
| 108 | 9 | Local variable `y2` is assigned to but never used |
| 109 | 9 | Local variable `corer2move2` is assigned to but never used |
| 110 | 9 | Local variable `move2corner` is assigned to but never used |
| 111 | 9 | Local variable `identity` is assigned to but never used |
| 127 | 9 | Local variable `Task` is assigned to but never used |
| 128 | 9 | Local variable `project_path` is assigned to but never used |
| 129 | 9 | Local variable `scorer` is assigned to but never used |
| 130 | 9 | Local variable `date` is assigned to but never used |
| 131 | 9 | Local variable `video_sets` is assigned to but never used |
| 132 | 9 | Local variable `individuals` is assigned to but never used |
| 133 | 9 | Local variable `multianimalbodyparts` is assigned to but never used |
| 134 | 9 | Local variable `skeleton` is assigned to but never used |
| 135 | 9 | Local variable `bodyparts` is assigned to but never used |
| 136 | 9 | Local variable `start` is assigned to but never used |
| 137 | 9 | Local variable `stop` is assigned to but never used |
| 138 | 9 | Local variable `numframes2pick` is assigned to but never used |
| 139 | 9 | Local variable `skeleton_color` is assigned to but never used |
| 140 | 9 | Local variable `pcutoff` is assigned to but never used |
| 141 | 9 | Local variable `dotsize` is assigned to but never used |
| 142 | 9 | Local variable `alphavalue` is assigned to but never used |
| 143 | 9 | Local variable `colormap` is assigned to but never used |
| 144 | 9 | Local variable `TrainingFraction` is assigned to but never used |
| 145 | 9 | Local variable `iteration` is assigned to but never used |
| 146 | 9 | Local variable `default_net_type` is assigned to but never used |
| 147 | 9 | Local variable `default_augmenter` is assigned to but never used |
| 148 | 9 | Local variable `snapshotindex` is assigned to but never used |
| 149 | 9 | Local variable `batch_size` is assigned to but never used |
| 150 | 9 | Local variable `cropping` is assigned to but never used |
| 151 | 9 | Local variable `croppedtraining` is assigned to but never used |
| 152 | 9 | Local variable `multianimalproject` is assigned to but never used |
| 153 | 9 | Local variable `uniquebodyparts` is assigned to but never used |
| 154 | 9 | Local variable `x1` is assigned to but never used |
| 155 | 9 | Local variable `x2` is assigned to but never used |
| 156 | 9 | Local variable `y1` is assigned to but never used |
| 157 | 9 | Local variable `y2` is assigned to but never used |
| 158 | 9 | Local variable `corer2move2` is assigned to but never used |
| 159 | 9 | Local variable `move2corner` is assigned to but never used |
| 160 | 9 | Local variable `identity` is assigned to but never used |
| 239 | 5 | Local variable `total_annotations` is assigned to but never used |
| 243 | 5 | Local variable `count` is assigned to but never used |
| 271 | 5 | Local variable `temp_count` is assigned to but never used |
| 375 | 5 | Local variable `nbodyparts` is assigned to but never used |
| 440 | 5 | Local variable `total_annotations` is assigned to but never used |
| 448 | 9 | Local variable `datasetname` is assigned to but never used |
| 488 | 9 | Local variable `freq` is assigned to but never used |
| 490 | 13 | Local variable `filename` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:80"
```

#### `deeplabcut\utils\pseudo_label.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 53 | 5 | Local variable `arranged_preds_list` is assigned to but never used |
| 101 | 5 | Local variable `fps` is assigned to but never used |
| 130 | 5 | Local variable `heatmap` is assigned to but never used |
| 401 | 5 | Local variable `new_predictions` is assigned to but never used |
| 403 | 5 | Local variable `num_kpts` is assigned to but never used |
| 447 | 13 | Local variable `bbox_confidence` is assigned to but never used |
| 474 | 5 | Local variable `test_annotations` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\utils\pseudo_label.py:53"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 129 | 9 | Local variable `num_kpts` is assigned to but never used |
| 315 | 9 | Local variable `num_images` is assigned to but never used |
| 705 | 13 | Local variable `j_x_sm` is assigned to but never used |
| 707 | 13 | Local variable `j_y_sm` is assigned to but never used |
| 709 | 13 | Local variable `map_j` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:129"
```

#### `deeplabcut\modelzoo\generalized_data_converter\utils.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 110 | 9 | Local variable `pickle_obj` is assigned to but never used |
| 125 | 5 | Local variable `video_name` is assigned to but never used |
| 149 | 5 | Local variable `bodyparts` is assigned to but never used |
| 289 | 5 | Local variable `visited` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\utils.py:110"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 305 | 13 | Local variable `scorer_cam1` is assigned to but never used |
| 306 | 13 | Local variable `scorer_cam2` is assigned to but never used |
| 308 | 13 | Local variable `bodyparts` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:305"
```

#### `deeplabcut\pose_estimation_pytorch\models\predictors\dekr_predictor.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 250 | 9 | Local variable `pool1` is assigned to but never used |
| 252 | 9 | Local variable `pool3` is assigned to but never used |
| 253 | 9 | Local variable `map_size` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\predictors\dekr_predictor.py:250"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 471 | 17 | Local variable `j_x_sm` is assigned to but never used |
| 473 | 17 | Local variable `j_y_sm` is assigned to but never used |
| 474 | 17 | Local variable `map_j` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:471"
```

#### `deeplabcut\pose_estimation_tensorflow\export.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 118 | 5 | Local variable `path_test_config` is assigned to but never used |
| 145 | 5 | Local variable `trainingsiterations` is assigned to but never used |
| 281 | 5 | Local variable `model_dir` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\export.py:118"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 183 | 5 | Local variable `pdindex` is assigned to but never used |
| 910 | 17 | Local variable `x0` is assigned to but never used |
| 910 | 21 | Local variable `y0` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:183"
```

#### `deeplabcut\pose_estimation_pytorch\models\necks\layers.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 176 | 9 | Local variable `b` is assigned to but never used |
| 176 | 12 | Local variable `n` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\necks\layers.py:176"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 420 | 9 | Local variable `mirror` is assigned to but never used |
| 426 | 9 | Local variable `im_file` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py:420"
```

#### `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 178 | 13 | Local variable `trainingsiterations` is assigned to but never used |
| 191 | 13 | Local variable `PredicteData` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\visualizemaps.py:178"
```

#### `tests\pose_estimation_pytorch\runners\bottum_up.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 57 | 5 | Local variable `template` is assigned to but never used |
| 86 | 5 | Local variable `runner` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\bottum_up.py:57"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 466 | 9 | Local variable `video_dir` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:466"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 951 | 5 | Local variable `dlc_root_path` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:951"
```

#### `deeplabcut\gui\tabs\modelzoo.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 459 | 17 | Local variable `results` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\modelzoo.py:459"
```

#### `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 131 | 9 | Local variable `super_bodyparts` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py:131"
```

#### `deeplabcut\modelzoo\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 226 | 5 | Local variable `available_projects` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\utils.py:226"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 414 | 13 | Local variable `norm` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:414"
```

#### `deeplabcut\pose_estimation_3d\plotting3D.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 133 | 5 | Local variable `start_path` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\plotting3D.py:133"
```

#### `deeplabcut\pose_estimation_pytorch\apis\videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 517 | 5 | Local variable `detector_path` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\videos.py:517"
```

#### `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 167 | 13 | Local variable `length` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\necks\transformer.py:167"
```

#### `deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 192 | 9 | Local variable `arranged_preds_list` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\modelzoo\memory_replay.py:192"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 53 | 14 | Local variable `ratio_w` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:53"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\multi.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 284 | 25 | Local variable `pre_stage_paf_output` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\multi.py:284"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 84 | 9 | Local variable `start` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_multianimal.py:84"
```

#### `deeplabcut\pose_estimation_tensorflow\training.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 198 | 13 | Local variable `supermodels` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\training.py:198"
```

#### `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 275 | 9 | Local variable `B` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py:275"
```

#### `deeplabcut\pose_tracking_pytorch\processor\processor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 271 | 5 | Local variable `val_loss` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\processor\processor.py:271"
```

#### `deeplabcut\pose_tracking_pytorch\train_dlctransreid.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 49 | 5 | Local variable `x_list` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\train_dlctransreid.py:49"
```

#### `deeplabcut\refine_training_dataset\outlier_frames.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 718 | 5 | Local variable `videofolder` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\outlier_frames.py:718"
```

#### `deeplabcut\utils\auxfun_videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 616 | 5 | Local variable `rs` is assigned to but never used |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_videos.py:616"
```

#### `examples\testscript_mobilenets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 75 | 5 | Local variable `DLC_config` is assigned to but never used |

Quick open commands:

```powershell
code -g "examples\testscript_mobilenets.py:75"
```

#### `tests\pose_estimation_pytorch\apis\test_apis_evaluate.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 228 | 5 | Local variable `num_unique` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\apis\test_apis_evaluate.py:228"
```

#### `tests\pose_estimation_pytorch\config\test_make_pose_config.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 338 | 43 | Local variable `err_info` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\config\test_make_pose_config.py:338"
```

#### `tests\pose_estimation_pytorch\data\test_transforms.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 99 | 5 | Local variable `aug` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_transforms.py:99"
```

#### `tests\pose_estimation_pytorch\modelzoo\test_load_superanimal_models.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 31 | 13 | Local variable `snapshot` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\modelzoo\test_load_superanimal_models.py:31"
```

#### `tests\pose_estimation_pytorch\other\test_api_utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 70 | 13 | Local variable `transformed` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\other\test_api_utils.py:70"
```

#### `tests\pose_estimation_pytorch\runners\test_runners_inference.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 32 | 9 | Local variable `runner` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\test_runners_inference.py:32"
```

#### `tests\test_auxiliaryfunctions.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 22 | 5 | Local variable `n_ext` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\test_auxiliaryfunctions.py:22"
```

#### `tests\test_pose_multianimal_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 109 | 9 | Local variable `batch` is assigned to but never used |

Quick open commands:

```powershell
code -g "tests\test_pose_multianimal_imgaug.py:109"
```

## E402

Count: **93**
Hint: Module import not at top of file. Move imports above executable code if possible.

### Files affected

| File | Count |
|---|---:|
| `docs\recipes\flip_and_rotate.ipynb` | 36 |
| `deeplabcut\pose_estimation_tensorflow\__init__.py` | 13 |
| `deeplabcut\__init__.py` | 12 |
| `deeplabcut\benchmark\metrics.py` | 8 |
| `deeplabcut\pose_estimation_tensorflow\core\train.py` | 6 |
| `testscript_cli.py` | 6 |
| `deeplabcut\pose_estimation_3d\plotting3D.py` | 5 |
| `examples\testscript_deterministicwithResNet152.py` | 4 |
| `examples\COLAB\COLAB_DEMO_mouse_openfield.ipynb` | 2 |
| `tests\pose_estimation_pytorch\modelzoo\test_fmpose_integration.py` | 1 |

### Details

#### `docs\recipes\flip_and_rotate.ipynb` (36)

| Line | Col | Message |
|---:|---:|---|
| 5 | 1 | Module level import not at top of cell |
| 7 | 1 | Module level import not at top of cell |
| 8 | 1 | Module level import not at top of cell |
| 8 | 1 | Module level import not at top of cell |
| 8 | 1 | Module level import not at top of cell |
| 8 | 1 | Module level import not at top of cell |
| 9 | 1 | Module level import not at top of cell |
| 9 | 1 | Module level import not at top of cell |
| 10 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |
| 12 | 1 | Module level import not at top of cell |
| 12 | 1 | Module level import not at top of cell |
| 13 | 1 | Module level import not at top of cell |
| 13 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 14 | 1 | Module level import not at top of cell |
| 16 | 1 | Module level import not at top of cell |
| 16 | 1 | Module level import not at top of cell |
| 16 | 1 | Module level import not at top of cell |
| 17 | 1 | Module level import not at top of cell |
| 17 | 1 | Module level import not at top of cell |
| 17 | 1 | Module level import not at top of cell |
| 17 | 1 | Module level import not at top of cell |
| 18 | 1 | Module level import not at top of cell |
| 18 | 1 | Module level import not at top of cell |
| 19 | 1 | Module level import not at top of cell |
| 20 | 1 | Module level import not at top of cell |
| 55 | 1 | Module level import not at top of cell |

Quick open commands:

```powershell
code -g "docs\recipes\flip_and_rotate.ipynb:5"
```

#### `deeplabcut\pose_estimation_tensorflow\__init__.py` (13)

| Line | Col | Message |
|---:|---:|---|
| 22 | 1 | Module level import not at top of file |
| 23 | 1 | Module level import not at top of file |
| 24 | 1 | Module level import not at top of file |
| 25 | 1 | Module level import not at top of file |
| 26 | 1 | Module level import not at top of file |
| 27 | 1 | Module level import not at top of file |
| 28 | 1 | Module level import not at top of file |
| 29 | 1 | Module level import not at top of file |
| 30 | 1 | Module level import not at top of file |
| 31 | 1 | Module level import not at top of file |
| 32 | 1 | Module level import not at top of file |
| 33 | 1 | Module level import not at top of file |
| 34 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\__init__.py:22"
```

#### `deeplabcut\__init__.py` (12)

| Line | Col | Message |
|---:|---:|---|
| 16 | 1 | Module level import not at top of file |
| 31 | 1 | Module level import not at top of file |
| 32 | 1 | Module level import not at top of file |
| 40 | 1 | Module level import not at top of file |
| 55 | 1 | Module level import not at top of file |
| 56 | 1 | Module level import not at top of file |
| 81 | 1 | Module level import not at top of file |
| 98 | 1 | Module level import not at top of file |
| 104 | 1 | Module level import not at top of file |
| 105 | 1 | Module level import not at top of file |
| 110 | 1 | Module level import not at top of file |
| 111 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "deeplabcut\__init__.py:16"
```

#### `deeplabcut\benchmark\metrics.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 23 | 1 | Module level import not at top of file |
| 24 | 1 | Module level import not at top of file |
| 25 | 1 | Module level import not at top of file |
| 27 | 1 | Module level import not at top of file |
| 28 | 1 | Module level import not at top of file |
| 30 | 1 | Module level import not at top of file |
| 31 | 1 | Module level import not at top of file |
| 32 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "deeplabcut\benchmark\metrics.py:23"
```

#### `deeplabcut\pose_estimation_tensorflow\core\train.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 25 | 1 | Module level import not at top of file |
| 27 | 1 | Module level import not at top of file |
| 28 | 1 | Module level import not at top of file |
| 32 | 1 | Module level import not at top of file |
| 33 | 1 | Module level import not at top of file |
| 34 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\train.py:25"
```

#### `testscript_cli.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 14 | 1 | Module level import not at top of file |
| 15 | 1 | Module level import not at top of file |
| 17 | 1 | Module level import not at top of file |
| 18 | 1 | Module level import not at top of file |
| 23 | 1 | Module level import not at top of file |
| 24 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "testscript_cli.py:14"
```

#### `deeplabcut\pose_estimation_3d\plotting3D.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 29 | 1 | Module level import not at top of file |
| 30 | 1 | Module level import not at top of file |
| 31 | 1 | Module level import not at top of file |
| 32 | 1 | Module level import not at top of file |
| 33 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\plotting3D.py:29"
```

#### `examples\testscript_deterministicwithResNet152.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 44 | 1 | Module level import not at top of file |
| 46 | 1 | Module level import not at top of file |
| 47 | 1 | Module level import not at top of file |
| 49 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "examples\testscript_deterministicwithResNet152.py:44"
```

#### `examples\COLAB\COLAB_DEMO_mouse_openfield.ipynb` (2)

| Line | Col | Message |
|---:|---:|---|
| 10 | 1 | Module level import not at top of cell |
| 11 | 1 | Module level import not at top of cell |

Quick open commands:

```powershell
code -g "examples\COLAB\COLAB_DEMO_mouse_openfield.ipynb:10"
```

#### `tests\pose_estimation_pytorch\modelzoo\test_fmpose_integration.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 19 | 1 | Module level import not at top of file |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\modelzoo\test_fmpose_integration.py:19"
```

## UP031

Count: **76**
Hint: Old `%` formatting. Convert to f-strings or `.format()` where appropriate.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_model.py` | 15 |
| `deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_builder.py` | 13 |
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 11 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 7 |
| `deeplabcut\pose_estimation_tensorflow\export.py` | 4 |
| `deeplabcut\pose_estimation_tensorflow\backbones\mobilenet.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\nnets\utils.py` | 3 |
| `deeplabcut\create_project\new.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` | 2 |
| `deeplabcut\create_project\add.py` | 1 |
| `deeplabcut\create_project\new_3d.py` | 1 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 1 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 1 |
| `deeplabcut\pose_estimation_3d\plotting3D.py` | 1 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` | 1 |
| `deeplabcut\post_processing\analyze_skeleton.py` | 1 |
| `deeplabcut\post_processing\filtering.py` | 1 |
| `deeplabcut\refine_training_dataset\outlier_frames.py` | 1 |
| `examples\COLAB\COLAB_DEMO_SuperAnimal.ipynb` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_model.py` (15)

| Line | Col | Message |
|---:|---:|---|
| 256 | 35 | Use format specifiers instead of percent format |
| 268 | 35 | Use format specifiers instead of percent format |
| 273 | 35 | Use format specifiers instead of percent format |
| 276 | 35 | Use format specifiers instead of percent format |
| 294 | 35 | Use format specifiers instead of percent format |
| 345 | 35 | Use format specifiers instead of percent format |
| 350 | 35 | Use format specifiers instead of percent format |
| 364 | 35 | Use format specifiers instead of percent format |
| 478 | 35 | Use format specifiers instead of percent format |
| 489 | 46 | Use format specifiers instead of percent format |
| 493 | 47 | Use format specifiers instead of percent format |
| 500 | 32 | Use format specifiers instead of percent format |
| 502 | 36 | Use format specifiers instead of percent format |
| 505 | 40 | Use format specifiers instead of percent format |
| 507 | 44 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_model.py:256"
```

#### `deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_builder.py` (13)

| Line | Col | Message |
|---:|---:|---|
| 77 | 13 | Use format specifiers instead of percent format |
| 78 | 13 | Use format specifiers instead of percent format |
| 79 | 13 | Use format specifiers instead of percent format |
| 80 | 13 | Use format specifiers instead of percent format |
| 81 | 13 | Use format specifiers instead of percent format |
| 82 | 13 | Use format specifiers instead of percent format |
| 83 | 13 | Use format specifiers instead of percent format |
| 86 | 25 | Use format specifiers instead of percent format |
| 190 | 35 | Use format specifiers instead of percent format |
| 242 | 43 | Use format specifiers instead of percent format |
| 243 | 25 | Use format specifiers instead of percent format |
| 244 | 25 | Use format specifiers instead of percent format |
| 245 | 25 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\backbones\efficientnet_builder.py:77"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (11)

| Line | Col | Message |
|---:|---:|---|
| 106 | 13 | Use format specifiers instead of percent format |
| 121 | 11 | Use format specifiers instead of percent format |
| 489 | 13 | Use format specifiers instead of percent format |
| 505 | 11 | Use format specifiers instead of percent format |
| 663 | 9 | Use format specifiers instead of percent format |
| 1077 | 13 | Use format specifiers instead of percent format |
| 1210 | 13 | Use format specifiers instead of percent format |
| 1225 | 11 | Use format specifiers instead of percent format |
| 1317 | 23 | Use format specifiers instead of percent format |
| 1548 | 13 | Use format specifiers instead of percent format |
| 1579 | 11 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:106"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (7)

| Line | Col | Message |
|---:|---:|---|
| 145 | 27 | Use format specifiers instead of percent format |
| 185 | 17 | Use format specifiers instead of percent format |
| 195 | 19 | Use format specifiers instead of percent format |
| 200 | 19 | Use format specifiers instead of percent format |
| 256 | 13 | Use format specifiers instead of percent format |
| 264 | 13 | Use format specifiers instead of percent format |
| 400 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:145"
```

#### `deeplabcut\pose_estimation_tensorflow\export.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 126 | 13 | Use format specifiers instead of percent format |
| 270 | 27 | Use format specifiers instead of percent format |
| 289 | 20 | Use format specifiers instead of percent format |
| 299 | 35 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\export.py:126"
```

#### `deeplabcut\pose_estimation_tensorflow\backbones\mobilenet.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 242 | 25 | Use format specifiers instead of percent format |
| 246 | 23 | Use format specifiers instead of percent format |
| 325 | 26 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\backbones\mobilenet.py:242"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\utils.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 115 | 23 | Use format specifiers instead of percent format |
| 118 | 35 | Use format specifiers instead of percent format |
| 162 | 21 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\utils.py:115"
```

#### `deeplabcut\create_project\new.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 202 | 43 | Use format specifiers instead of percent format |
| 306 | 9 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new.py:202"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 249 | 29 | Use format specifiers instead of percent format |
| 369 | 24 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py:249"
```

#### `deeplabcut\create_project\add.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 100 | 43 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\add.py:100"
```

#### `deeplabcut\create_project\new_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 126 | 9 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new_3d.py:126"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 363 | 23 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:363"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 332 | 11 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:332"
```

#### `deeplabcut\pose_estimation_3d\plotting3D.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 179 | 17 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\plotting3D.py:179"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 132 | 23 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:132"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 272 | 13 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:272"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 77 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:77"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 61 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:61"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 387 | 19 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:387"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 215 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_multianimal.py:215"
```

#### `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 142 | 17 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\visualizemaps.py:142"
```

#### `deeplabcut\post_processing\analyze_skeleton.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 263 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\analyze_skeleton.py:263"
```

#### `deeplabcut\post_processing\filtering.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 230 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\filtering.py:230"
```

#### `deeplabcut\refine_training_dataset\outlier_frames.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 931 | 15 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\outlier_frames.py:931"
```

#### `examples\COLAB\COLAB_DEMO_SuperAnimal.ipynb` (1)

| Line | Col | Message |
|---:|---:|---|
| 23 | 5 | Use format specifiers instead of percent format |

Quick open commands:

```powershell
code -g "examples\COLAB\COLAB_DEMO_SuperAnimal.ipynb:23"
```

## B007

Count: **51**
Hint: Unused loop variable. Rename to `_` or use it.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_tracking_pytorch\processor\processor.py` | 4 |
| `deeplabcut\core\trackingutils.py` | 3 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` | 3 |
| `deeplabcut\core\crossvalutils.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\apis\visualization.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 2 |
| `deeplabcut\utils\frameselectiontools.py` | 2 |
| `tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py` | 2 |
| `tests\pose_estimation_pytorch\runners\test_schedulers.py` | 2 |
| `deeplabcut\benchmark\utils.py` | 1 |
| `deeplabcut\core\inferenceutils.py` | 1 |
| `deeplabcut\core\metrics\matching.py` | 1 |
| `deeplabcut\gui\tabs\create_project.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\data\base.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\data\utils.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\modules\conv_module.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\modules\gated_attention_unit.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\dynamic_cropping.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` | 1 |
| `deeplabcut\refine_training_dataset\outlier_frames.py` | 1 |
| `deeplabcut\utils\visualization.py` | 1 |
| `examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb` | 1 |
| `tests\core\inferenceutils\test_map_computation.py` | 1 |
| `tests\core\metrics\test_metrics_map_computation.py` | 1 |
| `tests\create_project\test_video_set_configuration.py` | 1 |
| `tests\generate_training_dataset\test_trainset_metadata.py` | 1 |
| `tests\pose_estimation_pytorch\data\test_data_ctd.py` | 1 |
| `tests\pose_estimation_pytorch\other\test_api_utils.py` | 1 |
| `tests\test_auxfun_models.py` | 1 |
| `tests\test_auxiliaryfunctions.py` | 1 |

### Details

#### `deeplabcut\pose_tracking_pytorch\processor\processor.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 163 | 17 | Loop control variable `n_iter` not used within loop body |
| 218 | 9 | Loop control variable `n_iter` not used within loop body |
| 230 | 17 | Loop control variable `i` not used within loop body |
| 275 | 9 | Loop control variable `n_iter` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\processor\processor.py:163"
```

#### `deeplabcut\core\trackingutils.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 567 | 13 | Loop control variable `i` not used within loop body |
| 696 | 16 | Loop control variable `det` not used within loop body |
| 700 | 16 | Loop control variable `trk` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\core\trackingutils.py:567"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 100 | 13 | Loop control variable `dataset_name` not used within loop body |
| 172 | 13 | Loop control variable `k` not used within loop body |
| 191 | 13 | Loop control variable `dataset_name` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py:100"
```

#### `deeplabcut\core\crossvalutils.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 229 | 9 | Loop control variable `i` not used within loop body |
| 281 | 16 | Loop control variable `imname` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\core\crossvalutils.py:229"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 503 | 13 | Loop control variable `idx` not used within loop body |
| 639 | 21 | Loop control variable `kpt_name` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:503"
```

#### `deeplabcut\pose_estimation_pytorch\apis\visualization.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 149 | 17 | Loop control variable `idx` not used within loop body |
| 465 | 17 | Loop control variable `image_idx` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\visualization.py:149"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 135 | 13 | Loop control variable `image_id` not used within loop body |
| 546 | 17 | Loop control variable `k` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:135"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 163 | 21 | Loop control variable `scale_id` not used within loop body |
| 191 | 21 | Loop control variable `scale_id` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:163"
```

#### `deeplabcut\utils\frameselectiontools.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 291 | 30 | Loop control variable `index` not used within loop body |
| 305 | 30 | Loop control variable `index` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\utils\frameselectiontools.py:291"
```

#### `tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 166 | 10 | Loop control variable `start_1` not used within loop body |
| 166 | 37 | Loop control variable `end_2` not used within loop body |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\test_dynamic_cropper.py:166"
```

#### `tests\pose_estimation_pytorch\runners\test_schedulers.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 34 | 9 | Loop control variable `i` not used within loop body |
| 252 | 9 | Loop control variable `epoch` not used within loop body |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\runners\test_schedulers.py:34"
```

#### `deeplabcut\benchmark\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 66 | 9 | Loop control variable `loader` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\benchmark\utils.py:66"
```

#### `deeplabcut\core\inferenceutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 466 | 30 | Loop control variable `l` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\core\inferenceutils.py:466"
```

#### `deeplabcut\core\metrics\matching.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 102 | 14 | Loop control variable `pred` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\core\metrics\matching.py:102"
```

#### `deeplabcut\gui\tabs\create_project.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 134 | 17 | Loop control variable `entry` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\create_project.py:134"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 76 | 17 | Loop control variable `individual_id` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py:76"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 184 | 17 | Loop control variable `individual_id` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py:184"
```

#### `deeplabcut\pose_estimation_pytorch\data\base.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 337 | 17 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\base.py:337"
```

#### `deeplabcut\pose_estimation_pytorch\data\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 305 | 9 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\utils.py:305"
```

#### `deeplabcut\pose_estimation_pytorch\models\modules\conv_module.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 115 | 13 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\modules\conv_module.py:115"
```

#### `deeplabcut\pose_estimation_pytorch\models\modules\gated_attention_unit.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 42 | 9 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\modules\gated_attention_unit.py:42"
```

#### `deeplabcut\pose_estimation_pytorch\models\necks\transformer.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 201 | 13 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\necks\transformer.py:201"
```

#### `deeplabcut\pose_estimation_pytorch\runners\dynamic_cropping.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 528 | 13 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\dynamic_cropping.py:528"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 109 | 13 | Loop control variable `pi` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:109"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 59 | 9 | Loop control variable `n` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py:59"
```

#### `deeplabcut\refine_training_dataset\outlier_frames.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1096 | 9 | Loop control variable `findex` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\outlier_frames.py:1096"
```

#### `deeplabcut\utils\visualization.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 76 | 9 | Loop control variable `scorerindex` not used within loop body |

Quick open commands:

```powershell
code -g "deeplabcut\utils\visualization.py:76"
```

#### `examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb` (1)

| Line | Col | Message |
|---:|---:|---|
| 3 | 9 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "examples\COLAB\COLAB_BUCTD_and_CTD_tracking.ipynb:3"
```

#### `tests\core\inferenceutils\test_map_computation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 318 | 13 | Loop control variable `idv_id` not used within loop body |

Quick open commands:

```powershell
code -g "tests\core\inferenceutils\test_map_computation.py:318"
```

#### `tests\core\metrics\test_metrics_map_computation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 294 | 13 | Loop control variable `idv_id` not used within loop body |

Quick open commands:

```powershell
code -g "tests\core\metrics\test_metrics_map_computation.py:294"
```

#### `tests\create_project\test_video_set_configuration.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 255 | 9 | Loop control variable `video_path` not used within loop body |

Quick open commands:

```powershell
code -g "tests\create_project\test_video_set_configuration.py:255"
```

#### `tests\generate_training_dataset\test_trainset_metadata.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 81 | 9 | Loop control variable `name` not used within loop body |

Quick open commands:

```powershell
code -g "tests\generate_training_dataset\test_trainset_metadata.py:81"
```

#### `tests\pose_estimation_pytorch\data\test_data_ctd.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 173 | 25 | Loop control variable `img_index` not used within loop body |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\data\test_data_ctd.py:173"
```

#### `tests\pose_estimation_pytorch\other\test_api_utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 60 | 9 | Loop control variable `i` not used within loop body |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\other\test_api_utils.py:60"
```

#### `tests\test_auxfun_models.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 25 | 32 | Loop control variable `expected_path` not used within loop body |

Quick open commands:

```powershell
code -g "tests\test_auxfun_models.py:25"
```

#### `tests\test_auxiliaryfunctions.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 39 | 14 | Loop control variable `ext` not used within loop body |

Quick open commands:

```powershell
code -g "tests\test_auxiliaryfunctions.py:39"
```

## B028

Count: **49**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\utils\auxfun_videos.py` | 5 |
| `deeplabcut\core\inferenceutils.py` | 4 |
| `deeplabcut\pose_estimation_pytorch\data\cocoloader.py` | 4 |
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 3 |
| `deeplabcut\create_project\new.py` | 2 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 2 |
| `deeplabcut\gui\widgets.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` | 2 |
| `deeplabcut\modelzoo\utils.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\apis\tracklets.py` | 2 |
| `deeplabcut\pose_estimation_pytorch\data\transforms.py` | 2 |
| `deeplabcut\refine_training_dataset\stitch.py` | 2 |
| `deeplabcut\utils\skeleton.py` | 2 |
| `deeplabcut\__init__.py` | 1 |
| `deeplabcut\benchmark\base.py` | 1 |
| `deeplabcut\core\weight_init.py` | 1 |
| `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\base.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` | 1 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\modelzoo\utils.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\inference.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\snapshots.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\train.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\factory.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\nnets\factory.py` | 1 |
| `deeplabcut\utils\auxfun_multianimal.py` | 1 |
| `deeplabcut\utils\auxiliaryfunctions.py` | 1 |

### Details

#### `deeplabcut\utils\auxfun_videos.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 61 | 13 | No explicit `stacklevel` keyword argument found |
| 69 | 17 | No explicit `stacklevel` keyword argument found |
| 113 | 13 | No explicit `stacklevel` keyword argument found |
| 158 | 13 | No explicit `stacklevel` keyword argument found |
| 189 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_videos.py:61"
```

#### `deeplabcut\core\inferenceutils.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 263 | 13 | No explicit `stacklevel` keyword argument found |
| 343 | 13 | No explicit `stacklevel` keyword argument found |
| 351 | 13 | No explicit `stacklevel` keyword argument found |
| 367 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\core\inferenceutils.py:263"
```

#### `deeplabcut\pose_estimation_pytorch\data\cocoloader.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 145 | 17 | No explicit `stacklevel` keyword argument found |
| 152 | 13 | No explicit `stacklevel` keyword argument found |
| 203 | 13 | No explicit `stacklevel` keyword argument found |
| 223 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\cocoloader.py:145"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 824 | 13 | No explicit `stacklevel` keyword argument found |
| 1524 | 9 | No explicit `stacklevel` keyword argument found |
| 1561 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:824"
```

#### `deeplabcut\create_project\new.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 226 | 13 | No explicit `stacklevel` keyword argument found |
| 232 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new.py:226"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 944 | 9 | No explicit `stacklevel` keyword argument found |
| 1491 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:944"
```

#### `deeplabcut\gui\widgets.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 559 | 13 | No explicit `stacklevel` keyword argument found |
| 643 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\gui\widgets.py:559"
```

#### `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 24 | 13 | No explicit `stacklevel` keyword argument found |
| 122 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py:24"
```

#### `deeplabcut\modelzoo\utils.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 200 | 9 | No explicit `stacklevel` keyword argument found |
| 207 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\utils.py:200"
```

#### `deeplabcut\pose_estimation_pytorch\apis\tracklets.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 60 | 9 | No explicit `stacklevel` keyword argument found |
| 100 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\apis\tracklets.py:60"
```

#### `deeplabcut\pose_estimation_pytorch\data\transforms.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 52 | 13 | No explicit `stacklevel` keyword argument found |
| 422 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\transforms.py:52"
```

#### `deeplabcut\refine_training_dataset\stitch.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 671 | 13 | No explicit `stacklevel` keyword argument found |
| 726 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\stitch.py:671"
```

#### `deeplabcut\utils\skeleton.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 73 | 13 | No explicit `stacklevel` keyword argument found |
| 151 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\utils\skeleton.py:73"
```

#### `deeplabcut\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 73 | 5 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\__init__.py:73"
```

#### `deeplabcut\benchmark\base.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 120 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\benchmark\base.py:120"
```

#### `deeplabcut\core\weight_init.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 196 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\core\weight_init.py:196"
```

#### `deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 275 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\multiple_individuals_trainingsetmanipulation.py:275"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\base.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 193 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\base.py:193"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 121 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py:121"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 297 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:297"
```

#### `deeplabcut\pose_estimation_pytorch\modelzoo\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 178 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\modelzoo\utils.py:178"
```

#### `deeplabcut\pose_estimation_pytorch\runners\inference.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 242 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\inference.py:242"
```

#### `deeplabcut\pose_estimation_pytorch\runners\snapshots.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 137 | 13 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\snapshots.py:137"
```

#### `deeplabcut\pose_estimation_tensorflow\core\train.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 208 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\train.py:208"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\factory.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 26 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\factory.py:26"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\factory.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 21 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\factory.py:21"
```

#### `deeplabcut\utils\auxfun_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 81 | 17 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_multianimal.py:81"
```

#### `deeplabcut\utils\auxiliaryfunctions.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 292 | 9 | No explicit `stacklevel` keyword argument found |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions.py:292"
```

## F403

Count: **36**
Hint: `from x import *` makes names unclear. Replace with explicit imports.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\__init__.py` | 12 |
| `deeplabcut\utils\__init__.py` | 8 |
| `deeplabcut\generate_training_dataset\__init__.py` | 3 |
| `deeplabcut\pose_estimation_3d\__init__.py` | 3 |
| `deeplabcut\pose_tracking_pytorch\__init__.py` | 2 |
| `deeplabcut\refine_training_dataset\__init__.py` | 2 |
| `deeplabcut\gui\window.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\lib\crossvalutils.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\lib\inferenceutils.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\lib\trackingutils.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\util\__init__.py` | 1 |
| `deeplabcut\post_processing\__init__.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\__init__.py` (12)

| Line | Col | Message |
|---:|---:|---|
| 22 | 1 | `from deeplabcut.pose_estimation_tensorflow.config import *` used; unable to detect undefined names |
| 23 | 1 | `from deeplabcut.pose_estimation_tensorflow.core.evaluate import *` used; unable to detect undefined names |
| 24 | 1 | `from deeplabcut.pose_estimation_tensorflow.core.test import *` used; unable to detect undefined names |
| 25 | 1 | `from deeplabcut.pose_estimation_tensorflow.core.train import *` used; unable to detect undefined names |
| 26 | 1 | `from deeplabcut.pose_estimation_tensorflow.datasets import *` used; unable to detect undefined names |
| 27 | 1 | `from deeplabcut.pose_estimation_tensorflow.default_config import *` used; unable to detect undefined names |
| 29 | 1 | `from deeplabcut.pose_estimation_tensorflow.models import *` used; unable to detect undefined names |
| 30 | 1 | `from deeplabcut.pose_estimation_tensorflow.nnets import *` used; unable to detect undefined names |
| 31 | 1 | `from deeplabcut.pose_estimation_tensorflow.predict_videos import *` used; unable to detect undefined names |
| 32 | 1 | `from deeplabcut.pose_estimation_tensorflow.training import *` used; unable to detect undefined names |
| 33 | 1 | `from deeplabcut.pose_estimation_tensorflow.util import *` used; unable to detect undefined names |
| 34 | 1 | `from deeplabcut.pose_estimation_tensorflow.visualizemaps import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\__init__.py:22"
```

#### `deeplabcut\utils\__init__.py` (8)

| Line | Col | Message |
|---:|---:|---|
| 11 | 1 | `from deeplabcut.utils.auxfun_multianimal import *` used; unable to detect undefined names |
| 12 | 1 | `from deeplabcut.utils.auxfun_videos import *` used; unable to detect undefined names |
| 13 | 1 | `from deeplabcut.utils.auxiliaryfunctions import *` used; unable to detect undefined names |
| 14 | 1 | `from deeplabcut.utils.conversioncode import *` used; unable to detect undefined names |
| 15 | 1 | `from deeplabcut.utils.frameselectiontools import *` used; unable to detect undefined names |
| 16 | 1 | `from deeplabcut.utils.make_labeled_video import *` used; unable to detect undefined names |
| 17 | 1 | `from deeplabcut.utils.plotting import *` used; unable to detect undefined names |
| 18 | 1 | `from deeplabcut.utils.video_processor import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\utils\__init__.py:11"
```

#### `deeplabcut\generate_training_dataset\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from deeplabcut.generate_training_dataset.frame_extraction import *` used; unable to detect undefined names |
| 19 | 1 | `from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import *` used; unable to detect undefined names |
| 20 | 1 | `from deeplabcut.generate_training_dataset.trainingsetmanipulation import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\__init__.py:13"
```

#### `deeplabcut\pose_estimation_3d\__init__.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 11 | 1 | `from deeplabcut.pose_estimation_3d.camera_calibration import *` used; unable to detect undefined names |
| 12 | 1 | `from deeplabcut.pose_estimation_3d.plotting3D import *` used; unable to detect undefined names |
| 13 | 1 | `from deeplabcut.pose_estimation_3d.triangulation import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\__init__.py:11"
```

#### `deeplabcut\pose_tracking_pytorch\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from .create_dataset import *` used; unable to detect undefined names |
| 14 | 1 | `from .tracking_utils.preprocessing import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\__init__.py:13"
```

#### `deeplabcut\refine_training_dataset\__init__.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from deeplabcut.refine_training_dataset.outlier_frames import *` used; unable to detect undefined names |
| 14 | 1 | `from deeplabcut.refine_training_dataset.tracklets import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\__init__.py:13"
```

#### `deeplabcut\gui\window.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 40 | 1 | `from deeplabcut.gui.tabs import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\gui\window.py:40"
```

#### `deeplabcut\pose_estimation_tensorflow\lib\crossvalutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from deeplabcut.core.crossvalutils import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\lib\crossvalutils.py:13"
```

#### `deeplabcut\pose_estimation_tensorflow\lib\inferenceutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from deeplabcut.core.inferenceutils import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\lib\inferenceutils.py:13"
```

#### `deeplabcut\pose_estimation_tensorflow\lib\trackingutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 13 | 1 | `from deeplabcut.core.trackingutils import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\lib\trackingutils.py:13"
```

#### `deeplabcut\pose_estimation_tensorflow\util\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 19 | 1 | `from deeplabcut.pose_estimation_tensorflow.util.logging import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\util\__init__.py:19"
```

#### `deeplabcut\post_processing\__init__.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 22 | 1 | `from deeplabcut.post_processing.filtering import *` used; unable to detect undefined names |

Quick open commands:

```powershell
code -g "deeplabcut\post_processing\__init__.py:22"
```

## E712

Count: **22**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 4 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 3 |
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 2 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 2 |
| `deeplabcut\utils\auxfun_multianimal.py` | 2 |
| `tests\pose_estimation_pytorch\other\test_helper.py` | 2 |
| `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` | 1 |
| `deeplabcut\utils\auxiliaryfunctions_3d.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 282 | 8 | Avoid equality comparisons to `True`; use `rescale:` for truth checks |
| 369 | 20 | Avoid equality comparisons to `True`; use `show_errors:` for truth checks |
| 409 | 16 | Avoid equality comparisons to `True`; use `fulldata:` for truth checks |
| 430 | 12 | Avoid equality comparisons to `True`; use `fulldata:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:282"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 136 | 20 | Avoid equality comparisons to `True`; use `ret:` for truth checks |
| 163 | 8 | Avoid equality comparisons to `True`; use `calibrate:` for truth checks |
| 403 | 12 | Avoid equality comparisons to `True`; use `plot:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:136"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 1019 | 12 | Avoid equality comparisons to `True`; use `cfg["cropping"]:` for truth checks |
| 1268 | 8 | Avoid equality comparisons to `True`; use `os.path.isdir(directory):` for truth checks |
| 1297 | 20 | Avoid equality comparisons to `True`; use `cfg["cropping"]:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:1019"
```

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 187 | 12 | Avoid equality comparisons to `True`; use `dropped:` for truth checks |
| 679 | 8 | Avoid equality comparisons to `True`; use `uniform:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:187"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 249 | 12 | Avoid equality comparisons to `True`; use `append_image_id:` for truth checks |
| 457 | 12 | Avoid equality comparisons to `True`; use `append_image_id:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:249"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 107 | 8 | Avoid equality comparisons to `True`; use `isinstance(video_path, str):` for truth checks |
| 149 | 20 | Avoid equality comparisons to `True`; use `flag:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:107"
```

#### `deeplabcut\utils\auxfun_multianimal.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 281 | 12 | Avoid equality comparisons to `True`; use `userfeedback:` for truth checks |
| 366 | 12 | Avoid equality comparisons to `True`; use `userfeedback:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_multianimal.py:281"
```

#### `tests\pose_estimation_pytorch\other\test_helper.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 18 | 12 | Avoid equality comparisons to `True`; use `tmp_model.training:` for truth checks |
| 21 | 12 | Avoid equality comparisons to `False`; use `not tmp_model.training:` for false checks |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\other\test_helper.py:18"
```

#### `deeplabcut\pose_estimation_tensorflow\predict_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 194 | 12 | Avoid equality comparisons to `True`; use `cfg["cropping"]:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_multianimal.py:194"
```

#### `deeplabcut\utils\auxiliaryfunctions_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 94 | 8 | Avoid equality comparisons to `True`; use `plot:` for truth checks |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions_3d.py:94"
```

## F821

Count: **22**
Hint: Undefined name. Usually a real bug or missing import.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\utils\conversioncode.py` | 6 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` | 5 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 4 |
| `examples\JUPYTER\Demo_3D_DeepLabCut.ipynb` | 4 |
| `deeplabcut\pose_estimation_tensorflow\core\openvino\session.py` | 2 |
| `deeplabcut\pose_estimation_3d\triangulation.py` | 1 |

### Details

#### `deeplabcut\utils\conversioncode.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 112 | 11 | Undefined name `dlc` |
| 124 | 18 | Undefined name `tqdm` |
| 149 | 27 | Undefined name `np` |
| 152 | 29 | Undefined name `np` |
| 173 | 43 | Undefined name `np` |
| 184 | 43 | Undefined name `np` |

Quick open commands:

```powershell
code -g "deeplabcut\utils\conversioncode.py:112"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 218 | 38 | Undefined name `BasePoseDataset` |
| 219 | 37 | Undefined name `raw_2_imagename_with_id` |
| 220 | 37 | Undefined name `raw_2_imagename` |
| 222 | 36 | Undefined name `raw_2_imagename_with_id` |
| 223 | 36 | Undefined name `raw_2_imagename` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py:218"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (4)

| Line | Col | Message |
|---:|---:|---|
| 757 | 52 | Undefined name `x` |
| 757 | 86 | Undefined name `y` |
| 759 | 36 | Undefined name `y` |
| 759 | 70 | Undefined name `x` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:757"
```

#### `examples\JUPYTER\Demo_3D_DeepLabCut.ipynb` (4)

| Line | Col | Message |
|---:|---:|---|
| 1 | 30 | Undefined name `config_path3d` |
| 1 | 30 | Undefined name `config_path3d` |
| 4 | 31 | Undefined name `config_path3d` |
| 6 | 24 | Undefined name `config_path3d` |

Quick open commands:

```powershell
code -g "examples\JUPYTER\Demo_3D_DeepLabCut.ipynb:1"
```

#### `deeplabcut\pose_estimation_tensorflow\core\openvino\session.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 90 | 26 | Undefined name `out_id` |
| 107 | 18 | Undefined name `checkcropping` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\openvino\session.py:90"
```

#### `deeplabcut\pose_estimation_3d\triangulation.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 140 | 21 | Undefined name `warnings` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\triangulation.py:140"
```

## B904

Count: **19**
Hint: Inside `except`, use `raise ... from e` to preserve exception chaining.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 6 |
| `examples\testscript_3d.py` | 2 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 1 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\registry.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\schedulers.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\export.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\train_dlctransreid.py` | 1 |
| `deeplabcut\refine_training_dataset\stitch.py` | 1 |
| `deeplabcut\utils\auxfun_models.py` | 1 |
| `deeplabcut\utils\conversioncode.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (6)

| Line | Col | Message |
|---:|---:|---|
| 70 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 105 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 488 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 953 | 13 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 1209 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 1547 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:70"
```

#### `examples\testscript_3d.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 108 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |
| 126 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "examples\testscript_3d.py:108"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 470 | 13 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:470"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 158 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:158"
```

#### `deeplabcut\pose_estimation_pytorch\registry.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 69 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\registry.py:69"
```

#### `deeplabcut\pose_estimation_pytorch\runners\schedulers.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 117 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\schedulers.py:117"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 271 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:271"
```

#### `deeplabcut\pose_estimation_tensorflow\export.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 125 | 9 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\export.py:125"
```

#### `deeplabcut\pose_estimation_tensorflow\visualizemaps.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 141 | 13 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\visualizemaps.py:141"
```

#### `deeplabcut\pose_tracking_pytorch\train_dlctransreid.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 17 | 5 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\train_dlctransreid.py:17"
```

#### `deeplabcut\refine_training_dataset\stitch.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1162 | 17 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\stitch.py:1162"
```

#### `deeplabcut\utils\auxfun_models.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 177 | 13 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_models.py:177"
```

#### `deeplabcut\utils\conversioncode.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 303 | 13 | Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling |

Quick open commands:

```powershell
code -g "deeplabcut\utils\conversioncode.py:303"
```

## E722

Count: **19**
Hint: Bare `except:`. Catch `Exception` or a narrower exception type.

### Files affected

| File | Count |
|---|---:|
| `examples\testscript_3d.py` | 3 |
| `deeplabcut\pose_estimation_3d\camera_calibration.py` | 2 |
| `examples\testscript.py` | 2 |
| `deeplabcut\create_project\add.py` | 1 |
| `deeplabcut\create_project\new.py` | 1 |
| `deeplabcut\generate_training_dataset\frame_extraction.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\base.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\utils.py` | 1 |
| `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` | 1 |
| `deeplabcut\refine_training_dataset\outlier_frames.py` | 1 |
| `deeplabcut\utils\auxiliaryfunctions_3d.py` | 1 |
| `deeplabcut\utils\make_labeled_video.py` | 1 |

### Details

#### `examples\testscript_3d.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 92 | 5 | Do not use bare `except` |
| 107 | 5 | Do not use bare `except` |
| 125 | 5 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "examples\testscript_3d.py:92"
```

#### `deeplabcut\pose_estimation_3d\camera_calibration.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 97 | 5 | Do not use bare `except` |
| 157 | 5 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_3d\camera_calibration.py:97"
```

#### `examples\testscript.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 205 | 5 | Do not use bare `except` |
| 327 | 5 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "examples\testscript.py:205"
```

#### `deeplabcut\create_project\add.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 115 | 9 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\add.py:115"
```

#### `deeplabcut\create_project\new.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 219 | 9 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\create_project\new.py:219"
```

#### `deeplabcut\generate_training_dataset\frame_extraction.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 469 | 9 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\frame_extraction.py:469"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\base.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 221 | 13 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\base.py:221"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 80 | 17 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc.py:80"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 188 | 17 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py:188"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 472 | 13 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:472"
```

#### `deeplabcut\modelzoo\generalized_data_converter\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 32 | 5 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\utils.py:32"
```

#### `deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 329 | 13 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\model\backbones\vit_pytorch.py:329"
```

#### `deeplabcut\refine_training_dataset\outlier_frames.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 686 | 5 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\outlier_frames.py:686"
```

#### `deeplabcut\utils\auxiliaryfunctions_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 310 | 13 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxiliaryfunctions_3d.py:310"
```

#### `deeplabcut\utils\make_labeled_video.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1333 | 17 | Do not use bare `except` |

Quick open commands:

```powershell
code -g "deeplabcut\utils\make_labeled_video.py:1333"
```

## F405

Count: **16**
Hint: Likely consequence of `import *`. Import the name explicitly.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\gui\window.py` | 16 |

### Details

#### `deeplabcut\gui\window.py` (16)

| Line | Col | Message |
|---:|---:|---|
| 559 | 15 | `ProjectCreator` may be undefined, or defined from star imports |
| 563 | 24 | `OpenProject` may be undefined, or defined from star imports |
| 577 | 25 | `ModelZoo` may be undefined, or defined from star imports |
| 611 | 31 | `ManageProject` may be undefined, or defined from star imports |
| 612 | 31 | `ExtractFrames` may be undefined, or defined from star imports |
| 613 | 29 | `LabelFrames` may be undefined, or defined from star imports |
| 614 | 40 | `CreateTrainingDataset` may be undefined, or defined from star imports |
| 619 | 30 | `TrainNetwork` may be undefined, or defined from star imports |
| 624 | 33 | `EvaluateNetwork` may be undefined, or defined from star imports |
| 629 | 31 | `AnalyzeVideos` may be undefined, or defined from star imports |
| 630 | 41 | `UnsupervizedIdTracking` may be undefined, or defined from star imports |
| 635 | 30 | `CreateVideos` may be undefined, or defined from star imports |
| 640 | 39 | `ExtractOutlierFrames` may be undefined, or defined from star imports |
| 645 | 33 | `RefineTracklets` may be undefined, or defined from star imports |
| 646 | 25 | `ModelZoo` may be undefined, or defined from star imports |
| 647 | 29 | `VideoEditor` may be undefined, or defined from star imports |

Quick open commands:

```powershell
code -g "deeplabcut\gui\window.py:559"
```

## E721

Count: **14**
Hint: Avoid direct `type(x) == Y`; prefer `isinstance(x, Y)`.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 5 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 5 |
| `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` | 1 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\data\dlcloader.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 59 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 150 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 157 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 191 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 191 | 36 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:59"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (5)

| Line | Col | Message |
|---:|---:|---|
| 211 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 226 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 234 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 245 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |
| 245 | 36 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:211"
```

#### `deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 60 | 20 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\conversion_table\conversion_table.py:60"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 37 | 8 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\materialize.py:37"
```

#### `deeplabcut\pose_estimation_pytorch\data\dlcloader.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 322 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\dlcloader.py:322"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 169 | 16 | Use `is` and `is not` for type comparisons, or `isinstance()` for isinstance checks |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_tensorpack.py:169"
```

## B006

Count: **12**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` | 3 |
| `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` | 3 |
| `deeplabcut\utils\visualization.py` | 2 |
| `deeplabcut\modelzoo\fmpose_3d\fmpose3d.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` | 1 |
| `deeplabcut\utils\make_labeled_video.py` | 1 |

### Details

#### `deeplabcut\generate_training_dataset\trainingsetmanipulation.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 277 | 12 | Do not use mutable data structures for argument defaults |
| 1397 | 15 | Do not use mutable data structures for argument defaults |
| 1398 | 21 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\generate_training_dataset\trainingsetmanipulation.py:277"
```

#### `deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py` (3)

| Line | Col | Message |
|---:|---:|---|
| 125 | 16 | Do not use mutable data structures for argument defaults |
| 247 | 16 | Do not use mutable data structures for argument defaults |
| 426 | 16 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\modelzoo\api\superanimal_inference.py:125"
```

#### `deeplabcut\utils\visualization.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 60 | 12 | Do not use mutable data structures for argument defaults |
| 126 | 20 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\utils\visualization.py:60"
```

#### `deeplabcut\modelzoo\fmpose_3d\fmpose3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 21 | 27 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\fmpose_3d\fmpose3d.py:21"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 487 | 14 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate.py:487"
```

#### `deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 91 | 14 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\core\evaluate_multianimal.py:91"
```

#### `deeplabcut\utils\make_labeled_video.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 417 | 22 | Do not use mutable data structures for argument defaults |

Quick open commands:

```powershell
code -g "deeplabcut\utils\make_labeled_video.py:417"
```

## E711

Count: **7**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\modelzoo\generalized_data_converter\datasets\base_dlc.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py` | 2 |
| `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` | 1 |

### Details

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\base_dlc.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 25 | 29 | Comparison to `None` should be `cond is not None` |
| 25 | 54 | Comparison to `None` should be `cond is not None` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\base_dlc.py:25"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 84 | 29 | Comparison to `None` should be `cond is not None` |
| 84 | 54 | Comparison to `None` should be `cond is not None` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\ma_dlc_dataframe.py:84"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py` (2)

| Line | Col | Message |
|---:|---:|---|
| 85 | 29 | Comparison to `None` should be `cond is not None` |
| 85 | 54 | Comparison to `None` should be `cond is not None` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\single_dlc_dataframe.py:85"
```

#### `deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 43 | 30 | Comparison to `None` should be `cond is not None` |

Quick open commands:

```powershell
code -g "deeplabcut\modelzoo\generalized_data_converter\datasets\multi.py:43"
```

## E731

Count: **4**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` | 1 |
| `deeplabcut\refine_training_dataset\tracklets.py` | 1 |
| `deeplabcut\utils\auxfun_videos.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 144 | 9 | Do not assign a `lambda` expression, use a `def` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_imgaug.py:144"
```

#### `deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 181 | 9 | Do not assign a `lambda` expression, use a `def` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\datasets\pose_multianimal_imgaug.py:181"
```

#### `deeplabcut\refine_training_dataset\tracklets.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 88 | 9 | Do not assign a `lambda` expression, use a `def` |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\tracklets.py:88"
```

#### `deeplabcut\utils\auxfun_videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 272 | 9 | Do not assign a `lambda` expression, use a `def` |

Quick open commands:

```powershell
code -g "deeplabcut\utils\auxfun_videos.py:272"
```

## B008

Count: **3**
Hint: Function call in default arg. Use `None` + initialize inside the function.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\core\inferenceutils.py` | 1 |
| `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` | 1 |
| `examples\testscript_pytorch_single_animal.py` | 1 |

### Details

#### `deeplabcut\core\inferenceutils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1200 | 20 | Do not perform function call `np.linspace` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable |

Quick open commands:

```powershell
code -g "deeplabcut\core\inferenceutils.py:1200"
```

#### `deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 176 | 20 | Do not perform function call `expand_input_by_factor` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\conv_blocks.py:176"
```

#### `examples\testscript_pytorch_single_animal.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 29 | 57 | Do not perform function call `SyntheticProjectParameters` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable |

Quick open commands:

```powershell
code -g "examples\testscript_pytorch_single_animal.py:29"
```

## B023

Count: **2**
Hint: Function closes over loop variable. Bind it via default arg or helper.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\gui\tabs\train_network.py` | 1 |
| `deeplabcut\refine_training_dataset\stitch.py` | 1 |

### Details

#### `deeplabcut\gui\tabs\train_network.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 188 | 93 | Function definition does not bind loop variable `attribute` |

Quick open commands:

```powershell
code -g "deeplabcut\gui\tabs\train_network.py:188"
```

#### `deeplabcut\refine_training_dataset\stitch.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1177 | 32 | Function definition does not bind loop variable `stitcher` |

Quick open commands:

```powershell
code -g "deeplabcut\refine_training_dataset\stitch.py:1177"
```

## B024

Count: **2**
Hint: ABC without abstract method. Add `@abstractmethod` or remove ABC intent.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_pytorch\data\ctd.py` | 1 |
| `deeplabcut\pose_estimation_pytorch\runners\shelving.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_pytorch\data\ctd.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 26 | 7 | `CondProvider` is an abstract base class, but it has no abstract methods or properties |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\data\ctd.py:26"
```

#### `deeplabcut\pose_estimation_pytorch\runners\shelving.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 21 | 7 | `ShelfManager` is an abstract base class, but it has no abstract methods or properties |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\runners\shelving.py:21"
```

## F811

Count: **2**
Hint: Redefined while unused. Remove duplicate or rename.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_tracking_pytorch\processor\processor.py` | 1 |
| `tests\generate_training_dataset\test_trainset_metadata.py` | 1 |

### Details

#### `deeplabcut\pose_tracking_pytorch\processor\processor.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 26 | 5 | Redefinition of unused `dist` from line 19: `dist` redefined here |

Quick open commands:

```powershell
code -g "deeplabcut\pose_tracking_pytorch\processor\processor.py:26"
```

#### `tests\generate_training_dataset\test_trainset_metadata.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 246 | 5 | Redefinition of unused `test_add_shuffle` from line 210: `test_add_shuffle` redefined here |

Quick open commands:

```powershell
code -g "tests\generate_training_dataset\test_trainset_metadata.py:246"
```

## B011

Count: **1**

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\nnets\utils.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\nnets\utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 115 | 16 | Do not `assert False` (`python -O` removes these calls), raise `AssertionError()` |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\nnets\utils.py:115"
```

## B012

Count: **1**
Hint: Jump statement in `finally` can swallow exceptions. Restructure flow.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_tensorflow\predict_videos.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_tensorflow\predict_videos.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 1053 | 9 | `return` inside `finally` blocks cause exceptions to be silenced |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_tensorflow\predict_videos.py:1053"
```

## B016

Count: **1**
Hint: Raise an exception instance/class, not a literal.

### Files affected

| File | Count |
|---|---:|
| `examples\testscript_3d.py` | 1 |

### Details

#### `examples\testscript_3d.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 126 | 16 | Cannot raise a literal. Did you intend to return it or raise an Exception? |

Quick open commands:

```powershell
code -g "examples\testscript_3d.py:126"
```

## B017

Count: **1**
Hint: Use a more specific exception with `assertRaises`.

### Files affected

| File | Count |
|---|---:|
| `tests\pose_estimation_pytorch\other\test_api_utils.py` | 1 |

### Details

#### `tests\pose_estimation_pytorch\other\test_api_utils.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 67 | 14 | Do not assert blind exception: `Exception` |

Quick open commands:

```powershell
code -g "tests\pose_estimation_pytorch\other\test_api_utils.py:67"
```

## B020

Count: **1**
Hint: Loop variable overrides iterator. Rename loop variables.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 196 | 13 | Loop control variable `out_channels` overrides iterable it iterates |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\heads\simple_head.py:196"
```

## B027

Count: **1**
Hint: Empty method in ABC without abstract decorator. Add `@abstractmethod` or implement it.

### Files affected

| File | Count |
|---|---:|
| `deeplabcut\pose_estimation_pytorch\models\modules\kpt_encoders.py` | 1 |

### Details

#### `deeplabcut\pose_estimation_pytorch\models\modules\kpt_encoders.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 47 | 5 | `BaseKeypointEncoder.num_channels` is an empty method in an abstract base class, but has no abstract decorator |

Quick open commands:

```powershell
code -g "deeplabcut\pose_estimation_pytorch\models\modules\kpt_encoders.py:47"
```

## UP028

Count: **1**

### Files affected

| File | Count |
|---|---:|
| `tools\update_license_headers.py` | 1 |

### Details

#### `tools\update_license_headers.py` (1)

| Line | Col | Message |
|---:|---:|---|
| 32 | 13 | Replace `yield` over `for` loop with `yield from` |

Quick open commands:

```powershell
code -g "tools\update_license_headers.py:32"
```
