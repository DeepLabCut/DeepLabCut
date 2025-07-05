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
from __future__ import annotations

import os
from pathlib import Path
import re

import dlclibrary
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Slot

import deeplabcut
import deeplabcut.compat as compat
from deeplabcut.core.engine import Engine
from deeplabcut.core.weight_init import WeightInitialization
from deeplabcut.generate_training_dataset import get_existing_shuffle_indices
from deeplabcut.generate_training_dataset.metadata import get_shuffle_engine
from deeplabcut.gui.components import (
    DefaultTab,
    ShuffleSpinBox,
    ConditionsSelectionWidget,
    _create_grid_layout,
    _create_label_widget,
    set_combo_items,
    _create_message_box,
    _create_confirmation_box,
)
from deeplabcut.gui.displays.shuffle_metadata_viewer import ShuffleMetadataViewer
from deeplabcut.gui.dlc_params import DLCParams
from deeplabcut.gui.widgets import launch_napari
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.pose_estimation_pytorch import (
    available_models,
    is_model_top_down,
    is_model_cond_top_down,
)
from deeplabcut.utils.auxiliaryfunctions import (
    get_data_and_metadata_filenames,
    get_training_set_folder,
)


class CreateTrainingDataset(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(CreateTrainingDataset, self).__init__(root, parent, h1_description)

        self.model_comparison = False

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.mapping_button = QtWidgets.QPushButton("Edit Conversion Table")
        self.mapping_button.clicked.connect(self.edit_conversion_table)
        self.mapping_button.setVisible(False)
        self.root.engine_change.connect(self.set_edit_table_visibility)

        self.ok_button = QtWidgets.QPushButton("Create Training Dataset")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.create_training_dataset)

        self.main_layout.addWidget(self.mapping_button, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        self.view_shuffles_button = QtWidgets.QPushButton("View Existing Shuffles")
        self.view_shuffles_button.clicked.connect(self.view_shuffles)
        self.main_layout.addWidget(self.view_shuffles_button, alignment=Qt.AlignLeft)

        self.help_button = QtWidgets.QPushButton("Help")
        self.help_button.clicked.connect(self.show_help_dialog)
        self.main_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)

    def set_edit_table_visibility(self) -> None:
        has_conversion_tables = bool(
            self.root.cfg.get("SuperAnimalConversionTables", {})
        )
        is_pytorch_engine = self.root.engine == Engine.PYTORCH
        is_finetuning = self.weight_init_selector.with_decoder
        self.mapping_button.setVisible(
            has_conversion_tables & is_pytorch_engine & is_finetuning
        )

    def show_help_dialog(self):
        dialog = QtWidgets.QDialog(self)
        layout = QtWidgets.QVBoxLayout()
        if self.root.is_multianimal:
            func = deeplabcut.create_multianimaltraining_dataset
        else:
            func = deeplabcut.create_training_dataset
        label = QtWidgets.QLabel(func.__doc__, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        dialog.setLayout(layout)
        dialog.exec_()

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        # Shuffle
        shuffle_label = QtWidgets.QLabel("Shuffle")
        self.shuffle = ShuffleSpinBox(root=self.root, parent=self)

        # Dataset choices
        self.weight_init_label = QtWidgets.QLabel("Weight Initialization")
        self.weight_init_selector = WeightInitializationSelector(self.root)
        self.update_weight_init_methods(self.root.engine)
        self.root.engine_change.connect(self.update_weight_init_methods)

        # Augmentation method
        augmentation_label = QtWidgets.QLabel("Augmentation method")
        self.aug_choice = QtWidgets.QComboBox()
        self.update_aug_methods(self.root.engine)
        self.root.engine_change.connect(self.update_aug_methods)
        self.aug_choice.currentTextChanged.connect(self.log_augmentation_choice)

        # Neural Network
        nnet_label = QtWidgets.QLabel("Network architecture")
        self.net_choice = QtWidgets.QComboBox()
        self.net_choice.setMinimumWidth(200)
        self.update_nets(self.root.engine)
        self.root.engine_change.connect(self.update_nets)
        self.net_choice.currentTextChanged.connect(self.log_net_choice)

        # Update Net types when selected weight init changes
        self.weight_init_selector.weight_init_choice.currentTextChanged.connect(
            lambda _: self.update_nets(None)
        )
        self.weight_init_selector.weight_init_choice.currentTextChanged.connect(
            lambda _: self.set_edit_table_visibility()
        )

        # Detector selection for top-down models
        self.detector_label = QtWidgets.QLabel("Detector architecture")
        self.detector_choice = QtWidgets.QComboBox()
        self.detector_choice.setMinimumWidth(200)
        self.update_detectors(engine=self.root.engine)
        self.root.engine_change.connect(
            lambda engine: self.update_detectors(engine=engine)
        )
        self.net_choice.currentTextChanged.connect(
            lambda new_net_choice: self.update_detectors(net_choice=new_net_choice)
        )

        # Conditions selection for CTD models
        self.conditions_label = QtWidgets.QLabel("Conditions")
        self.conditions_selection_widget = ConditionsSelectionWidget(
            root=self.root, parent=self
        )
        self.update_conditions(engine=self.root.engine)
        self.root.engine_change.connect(
            lambda engine: self.update_conditions(engine=engine)
        )
        self.net_choice.currentTextChanged.connect(
            lambda new_net_choice: self.update_conditions(
                engine=self.root.engine, net_choice=new_net_choice
            )
        )

        # Overwrite selection
        self.overwrite = QtWidgets.QCheckBox("Overwrite if exists")
        self.overwrite.setChecked(False)
        self.overwrite.setToolTip(
            "When checked, creating a new shuffle with an index that already exists "
            "will overwrite the existing index. Be careful with this option as you "
            "might lose data."
        )
        self.overwrite.stateChanged.connect(
            lambda s: self.root.logger.info(f"Overwrite: {s}")
        )

        # Use same data split as another shuffle
        self.data_split_selection = DataSplitSelector(self.root, self)

        layout.addWidget(shuffle_label, 0, 0)
        layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(self.weight_init_label, 0, 2)
        layout.addWidget(self.weight_init_selector, 0, 3)

        layout.addWidget(nnet_label, 1, 0)
        layout.addWidget(self.net_choice, 1, 1)
        layout.addWidget(augmentation_label, 1, 2)
        layout.addWidget(self.aug_choice, 1, 3)

        layout.addWidget(self.detector_label, 2, 0)
        layout.addWidget(self.detector_choice, 2, 1)

        layout.addWidget(self.conditions_label, 3, 0)
        layout.addWidget(self.conditions_selection_widget, 3, 1)

        layout.addWidget(self.overwrite, 4, 0)
        layout.addWidget(self.data_split_selection, 5, 0)

    def log_net_choice(self, net):
        self.root.logger.info(f"Network architecture set to {net.upper()}")

    def log_augmentation_choice(self, augmentation):
        self.root.logger.info(f"Image augmentation set to {augmentation.upper()}")

    def edit_conversion_table(self):
        # Test beforehand whether a conversion table exists
        memory_replay_folder = Path(self.root.project_folder) / "memory_replay"
        conversion_matrix_out_path = str(memory_replay_folder / "confusion_matrix.png")
        files = [self.root.config]
        if os.path.exists(conversion_matrix_out_path):
            files.append(conversion_matrix_out_path)
        _ = launch_napari(files)

    def create_training_dataset(self):
        shuffle = self.shuffle.value()
        cfg = self.root.cfg
        existing_indices = get_existing_shuffle_indices(
            cfg=cfg, train_fraction=cfg["TrainingFraction"][self.root.trainingset_index]
        )

        overwrite = self.overwrite.isChecked()
        if shuffle in existing_indices:
            if overwrite:
                if not self._confirm_overwrite(shuffle, existing_indices):
                    return
            else:
                msg = _create_message_box(
                    f"The training dataset could not be created.",
                    (
                        f"Shuffle {shuffle} already exists - you can create a new "
                        "training dataset with an unused shuffle index (existing "
                        f"shuffles are {existing_indices}) or you can overwrite the "
                        f"shuffle by ticking the 'Overwrite' checkbox"
                    ),
                )
                msg.exec_()
                self.root.writer.write("Training dataset creation failed.")
                return

        if self.model_comparison:
            raise NotImplementedError
            # TODO: finish model_comparison
            # deeplabcut.create_training_model_comparison(
            #     config_file,
            #     num_shuffles=shuffle,
            #     net_types=self.net_type,
            #     augmenter_types=self.aug_type,
            # )
        else:
            try:
                engine = self.root.engine
                net_type = self.net_choice.currentText()
                detector_type = None
                ctd_conditions = None
                if engine == Engine.TF:
                    import tensorflow

                    # try importing TF so they can't create shuffles for it if they
                    # don't have it installed
                elif engine == Engine.PYTORCH:
                    if is_model_top_down(net_type):
                        detector_type = self.detector_choice.currentText()
                    elif is_model_cond_top_down(net_type):
                        ctd_conditions = self._build_ctd_conditions(
                            self.conditions_selection_widget.selected_conditions
                        )

                try:
                    weight_init = (
                        self.weight_init_selector.get_super_animal_weight_init(
                            net_type,
                            detector_type,
                        )
                    )
                except ValueError as err:
                    print(f"The training dataset could not be created: {err}.")
                    return

                if self.data_split_selection.selected:
                    deeplabcut.create_training_dataset_from_existing_split(
                        self.root.config,
                        from_shuffle=self.data_split_selection.from_shuffle,
                        shuffles=[self.shuffle.value()],
                        net_type=net_type,
                        detector_type=detector_type,
                        userfeedback=not overwrite,
                        weight_init=weight_init,
                        engine=engine,
                        ctd_conditions=ctd_conditions,
                    )

                elif self.root.is_multianimal:
                    deeplabcut.create_multianimaltraining_dataset(
                        self.root.config,
                        shuffle,
                        Shuffles=[self.shuffle.value()],
                        net_type=net_type,
                        detector_type=detector_type,
                        userfeedback=not overwrite,
                        weight_init=weight_init,
                        engine=engine,
                        ctd_conditions=ctd_conditions,
                    )
                else:
                    deeplabcut.create_training_dataset(
                        self.root.config,
                        shuffle,
                        Shuffles=[self.shuffle.value()],
                        net_type=net_type,
                        detector_type=detector_type,
                        augmenter_type=self.aug_choice.currentText(),
                        userfeedback=not overwrite,
                        weight_init=weight_init,
                        engine=engine,
                        ctd_conditions=ctd_conditions,
                    )
            except ValueError as err:
                msg = _create_message_box(
                    f"The training dataset could not be created.",
                    str(err),
                )
                msg.exec_()
                return
            except ModuleNotFoundError as err:
                info_text = (
                    f"Error `{err}`. If the error is `ModuleNotFoundError: No module "
                    "named 'tensorflow'`, this is because you tried creating a "
                    "TensorFlow shuffle, but TensorFlow is not installed in your "
                    "environment. To create TensorFlow shuffles (and use TensorFlow "
                    "models), install it with\n"
                    "    Windows/Linux:\n"
                    "      pip install 'deeplabcut[tf]'\n"
                    "    Apple Silicon:\n"
                    "      pip install 'deeplabcut[apple_mchips]'"
                )
                msg = _create_message_box(
                    f"The training dataset could not be created.", info_text
                )
                msg.exec_()
                return

            # Check that training data files were indeed created.
            trainingsetfolder = get_training_set_folder(self.root.cfg)
            filenames = list(
                get_data_and_metadata_filenames(
                    trainingsetfolder,
                    self.root.cfg["TrainingFraction"][0],
                    self.shuffle.value(),
                    self.root.cfg,
                )
            )
            if self.root.is_multianimal:
                filenames[0] = filenames[0].replace("mat", "pickle")
            if all(
                os.path.exists(os.path.join(self.root.project_folder, file))
                for file in filenames
            ):
                self.root.shuffle_created.emit(self.shuffle.value())
                msg = _create_message_box(
                    "The training dataset is successfully created.",
                    "Use the function 'train_network' to start training. Happy training!",
                )
                msg.exec_()
                self.root.writer.write("Training dataset successfully created.")
            else:
                msg = _create_message_box(
                    "The training dataset could not be created.",
                    "Make sure there are annotated data under labeled-data.",
                )
                msg.exec_()
                self.root.writer.write("Training dataset creation failed.")

    def _confirm_overwrite(self, shuffle: int, existing_indices: list[int]) -> bool:
        """
        Asks the user to confirm that they want to overwrite a shuffle.

        Args:
            shuffle: the shuffle the user wants to overwrite
            existing_indices: the indices of existing shuffles

        Returns:
            whether the user confirmed overwriting the shuffle
        """
        try:
            engine = get_shuffle_engine(
                self.root.cfg, self.root.trainingset_index, shuffle
            )
            engine_str = f" (with engine '{engine.aliases[0]}')"
        except ValueError:
            engine_str = ""

        conf = _create_confirmation_box(
            title=f"Are you sure you want to overwrite shuffle {shuffle}?",
            description=(
                f"As shuffle {shuffle} already exists{engine_str}, "
                f"the training-dataset files would be overwritten."
            ),
        )
        result = conf.exec()
        if result != QtWidgets.QMessageBox.Yes:
            msg = _create_message_box(
                text="The training dataset was not be created.",
                info_text=(
                    "You can create a shuffle with another index. Existing indices "
                    f"are {existing_indices}"
                ),
            )
            msg.exec_()
            self.root.writer.write("Training dataset creation interrupted.")
            return False

        return True

    def _build_ctd_conditions(
        self, conditions_path: str | Path
    ) -> Path | tuple[int, str]:
        """
        Builds CTD conditions in appropriate format from path to conditions.
        Args:
            conditions_path: str | Path:
                 path to conditions (path to snapshot or to predictions)

        Returns:
            ctd_conditions: Path | tuple[int, str]
                ctd conditions in the right format for deeplabcut.create_training_dataset() API method.

        Raises:
            Value error if conditions are missing or invalid.
        """
        if conditions_path is None:
            raise ValueError("No conditions were selected for CTD model.")
        else:
            conditions_path = Path(conditions_path)
            if conditions_path.suffix.lower() in [".h5", ".json"]:
                return conditions_path
            elif conditions_path.suffix.lower() == ".pt":
                match = re.search(r"shuffle(\d+)", str(conditions_path))
                if match:
                    shuffle_number = int(match.group(1))
                else:
                    raise ValueError("Shuffle number could not be extracted from path.")
                snapshot_filename = conditions_path.name
                return shuffle_number, snapshot_filename
            else:
                raise ValueError("Unsupported conditions file type")

    @Slot(Engine)
    def update_nets(self, engine: Engine | None) -> None:
        if engine is None:
            engine = self.root.engine

        default_net = None
        if engine == Engine.TF:
            nets = DLCParams.NNETS.copy()
            if not self.root.is_multianimal:
                nets.remove("dlcrnet_ms5")
        else:
            nets = available_models()
            net_filter = self.get_net_filter()
            default_net = self.get_default_net()
            td_prefix = "top_down_"
            if net_filter is not None:
                nets = [
                    n
                    for n in nets
                    if (
                        n in net_filter
                        or (
                            n.startswith(td_prefix)
                            and n[len(td_prefix) :] in net_filter
                        )
                    )
                ]

        if default_net is None:
            default_net = self.root.cfg.get("default_net_type", "resnet_50")
        if (
            engine == Engine.TF
            and default_net not in DLCParams.NNETS
            or engine == Engine.PYTORCH
            and default_net not in available_models()
        ):
            default_net = "resnet_50"

        set_combo_items(
            combo_box=self.net_choice,
            items=nets,
            index=nets.index(default_net) if default_net in nets else 0,
        )

    @Slot(Engine)
    def update_detectors(
        self,
        engine: Engine | None = None,
        net_choice: str | None = None,
    ) -> None:
        if engine is None:
            engine = self.root.engine

        if engine == Engine.TF:
            detectors = []
        else:
            # FIXME: Circular imports make it impossible to import this at the top
            from deeplabcut.pose_estimation_pytorch import available_detectors

            detectors = available_detectors()
            det_filter = self.get_detector_filter()
            if det_filter is not None:
                detectors = [d for d in detectors if d in det_filter]

        default_detector = self.get_default_detector()
        try:
            index = detectors.index(default_detector)
        except ValueError:
            try:
                index = detectors.index("ssdlite")
            except ValueError:
                index = -1
        set_combo_items(
            combo_box=self.detector_choice,
            items=detectors,
            index=index,
        )

        if net_choice is None:
            net_choice = self.net_choice.currentText()

        if engine == Engine.PYTORCH and is_model_top_down(net_choice):
            self.detector_label.show()
            self.detector_choice.show()
        else:
            self.detector_label.hide()
            self.detector_choice.hide()

    @Slot(Engine)
    def update_conditions(
        self,
        engine: Engine | None = None,
        net_choice: str | None = None,
    ) -> None:
        if engine is None:
            engine = self.root.engine

        if net_choice is None:
            net_choice = self.net_choice.currentText()

        if engine == Engine.PYTORCH and is_model_cond_top_down(net_choice):
            self.conditions_label.show()
            self.conditions_selection_widget.show()
        else:
            self.conditions_label.hide()
            self.conditions_selection_widget.hide()

    @Slot(Engine)
    def update_aug_methods(self, engine: Engine) -> None:
        methods = compat.get_available_aug_methods(engine)
        set_combo_items(
            combo_box=self.aug_choice,
            items=methods,
            index=0,
        )

    @Slot(Engine)
    def update_weight_init_methods(self, engine: Engine) -> None:
        if engine != Engine.PYTORCH:
            self.weight_init_label.hide()
            self.weight_init_selector.hide()
            return

        self.weight_init_label.show()
        self.weight_init_selector.update_choices(list(_WEIGHT_INIT_OPTIONS.keys()))
        self.weight_init_selector.show()

    def get_net_filter(self) -> list[str] | None:
        """Returns: the net type that can be used based on weight initialization"""
        if self.root.engine != Engine.PYTORCH:
            return None

        if self.weight_init_selector.weight_init not in _WEIGHT_INIT_OPTIONS:
            return None

        weight_init_cfg = _WEIGHT_INIT_OPTIONS[self.weight_init_selector.weight_init]
        if "super_animal" in weight_init_cfg:
            return dlclibrary.get_available_models(weight_init_cfg["super_animal"])

        return None

    def get_detector_filter(self) -> list[str] | None:
        """Returns: the detectors that can be used based on weight initialization"""
        if self.root.engine != Engine.PYTORCH:
            return None

        if self.weight_init_selector.weight_init not in _WEIGHT_INIT_OPTIONS:
            return None

        weight_init_cfg = _WEIGHT_INIT_OPTIONS[self.weight_init_selector.weight_init]
        if "super_animal" in weight_init_cfg:
            return dlclibrary.get_available_detectors(weight_init_cfg["super_animal"])

        return None

    def get_default_net(self) -> str | None:
        """Returns: the net type that can be used based on weight initialization"""
        if self.root.engine != Engine.PYTORCH:
            return None

        if self.weight_init_selector.weight_init not in _WEIGHT_INIT_OPTIONS:
            return None

        weight_init_cfg = _WEIGHT_INIT_OPTIONS[self.weight_init_selector.weight_init]
        return weight_init_cfg.get("default_net")

    def get_default_detector(self) -> str | None:
        """Returns: the detector type that can be used based on weight initialization"""
        if self.root.engine != Engine.PYTORCH:
            return None

        if self.weight_init_selector.weight_init not in _WEIGHT_INIT_OPTIONS:
            return None

        weight_init_cfg = _WEIGHT_INIT_OPTIONS[self.weight_init_selector.weight_init]
        return weight_init_cfg.get("default_detector")

    def view_shuffles(self) -> None:
        viewer = ShuffleMetadataViewer(root=self.root, parent=self)
        viewer.show()


class WeightInitializationSelector(QtWidgets.QWidget):
    """Widget to select weight initialization"""

    def __init__(self, root):
        super().__init__()
        self.root = root

        self.weight_init_choice = QtWidgets.QComboBox()

        self.memory_replay_label = QtWidgets.QLabel("With memory replay")
        self.memory_replay_box = QtWidgets.QCheckBox()
        self.memory_replay_label.hide()
        self.memory_replay_box.hide()

        memory_replay_layout = QtWidgets.QHBoxLayout()
        memory_replay_layout.addWidget(self.memory_replay_label)
        memory_replay_layout.addWidget(self.memory_replay_box)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.weight_init_choice)
        layout.addLayout(memory_replay_layout)
        self.setLayout(layout)

        self.weight_init_choice.currentTextChanged.connect(self._choice_changed)

    @property
    def weight_init(self) -> str:
        return self.weight_init_choice.currentText()

    @property
    def with_decoder(self) -> bool:
        weight_init_choice = self.weight_init_choice.currentText()
        return "fine-tuning" in weight_init_choice.lower()

    @property
    def memory_replay(self) -> bool:
        return self.memory_replay_box.isChecked()

    def update_choices(self, choices: list[str]) -> None:
        """Updates the WeightInitialization methods that can be selected"""
        set_combo_items(
            combo_box=self.weight_init_choice,
            items=choices,
        )

    def get_super_animal_weight_init(
        self,
        net_type: str,
        detector_type: str,
    ) -> WeightInitialization | None:
        """
        Args:
            net_type: The architecture of the pose model from which to fine-tune a
                SuperAnimal model.
            detector_type: The architecture of the detector from which to fine-tune a
                SuperAnimal model.

        Raises:
            ValueError if WeightInitialization should be defined but could not be
                created (e.g. if there's no conversion table).
        """
        if self.root.engine != Engine.PYTORCH:
            return None

        weight_init_choice = self.weight_init_choice.currentText()
        if "imagenet" in weight_init_choice.lower():
            return

        weight_init_data = _WEIGHT_INIT_OPTIONS[weight_init_choice]
        super_animal = weight_init_data["super_animal"]
        if net_type.startswith("top_down_"):
            net_type = net_type[len("top_down_") :]
        try:
            weight_init = build_weight_init(
                self.root.cfg,
                super_animal=super_animal,
                model_name=net_type,
                detector_name=detector_type,
                with_decoder=self.with_decoder,
                memory_replay=self.memory_replay,
            )
        except ValueError as err:
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                (
                    f"No Conversion table specified for {super_animal} in the project "
                    "configuration file. Please create a conversion table using the GUI"
                    ", with ``deeplabcut.modelzoo.utils.create_conversion_table``, or "
                    "by adding it to your project's configuration file manually."
                ),
            )
            raise err

        return weight_init

    def _choice_changed(self, state: str) -> None:
        if "fine-tuning" in str(state).lower():
            self.memory_replay_label.show()
            self.memory_replay_box.show()
        else:
            self.memory_replay_label.hide()
            self.memory_replay_box.hide()


class DataSplitSelector(QtWidgets.QWidget):
    """Allows users to create training sets with the same train/test split as another"""

    def __init__(self, root: QtWidgets.QMainWindow, parent: QtWidgets.QWidget):
        super().__init__()
        self.root = root
        self.parent = parent

        self.setToolTip(
            "This allows you to create a shuffle where the data split is the same as "
            "one of your existing shuffles (the images on which the model is "
            "trained/tested are the same)."
        )

        layout = QtWidgets.QVBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        box_layout = QtWidgets.QHBoxLayout()
        box_layout.setSpacing(0)
        box_layout.setContentsMargins(0, 0, 0, 0)

        selector_layout = QtWidgets.QHBoxLayout()
        selector_layout.setSpacing(0)
        selector_layout.setContentsMargins(0, 0, 0, 0)

        self.shuffle_label = QtWidgets.QLabel("From shuffle:")
        self.shuffle_label.hide()
        self.shuffle_selector = QtWidgets.QSpinBox()
        self.shuffle_selector.setMaximum(10_000)
        self.shuffle_selector.setValue(0)
        self.shuffle_selector.hide()

        self.box = QtWidgets.QCheckBox(parent=self)
        self.box.stateChanged.connect(self._checkbox_status_changed)
        self.box_label = QtWidgets.QLabel("Use an existing data split")

        box_layout.addWidget(self.box)
        box_layout.addWidget(self.box_label)
        selector_layout.addWidget(self.shuffle_label)
        selector_layout.addWidget(self.shuffle_selector)
        layout.addLayout(box_layout)
        layout.addLayout(selector_layout)
        self.setLayout(layout)

    @property
    def selected(self) -> bool:
        return self.box.isChecked()

    @property
    def from_shuffle(self) -> int:
        """The shuffle from which to copy the data split"""
        return self.shuffle_selector.value()

    def _checkbox_status_changed(self, state: int) -> None:
        if Qt.CheckState(state) == Qt.Checked:
            self.shuffle_selector.show()
            self.shuffle_label.show()
        else:
            self.shuffle_selector.hide()
            self.shuffle_label.hide()


_WEIGHT_INIT_OPTIONS = {  # FIXME - Generate dynamically
    "Transfer Learning - ImageNet": {
        "model_filter": None,
        "detector_filter": None,
    },
    "Transfer Learning - SuperAnimal Bird": {
        "default_net": "top_down_resnet_50",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_bird",
    },
    "Transfer Learning - SuperAnimal Quadruped": {
        "default_net": "top_down_hrnet_w32",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_quadruped",
    },
    "Transfer Learning - SuperAnimal TopViewMouse": {
        "default_net": "top_down_hrnet_w32",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_topviewmouse",
    },
    "Fine-tuning - SuperAnimal Bird": {
        "default_net": "top_down_resnet_50",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_bird",
    },
    "Fine-tuning - SuperAnimal Quadruped": {
        "default_net": "top_down_hrnet_w32",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_quadruped",
    },
    "Fine-tuning - SuperAnimal TopViewMouse": {
        "default_net": "top_down_hrnet_w32",
        "default_detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "super_animal": "superanimal_topviewmouse",
    },
}
