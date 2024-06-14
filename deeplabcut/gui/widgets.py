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
import ast
import os
import warnings

import matplotlib.colors as mcolors
import napari
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, Button, LassoSelector
from queue import Queue
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QStandardItemModel, QStandardItem, QCursor, QAction
from scipy.spatial import cKDTree as KDTree
from skimage import io

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.auxfun_videos import VideoWriter


def launch_napari(files=None, plugin="napari-deeplabcut", stack=False):
    viewer = napari.Viewer()
    if plugin == "napari-deeplabcut":
        # Automatically activate the napari-deeplabcut plugin
        for action in viewer.window.plugins_menu.actions():
            if "deeplabcut" in action.text():
                action.trigger()
                break
    if files is not None:
        viewer.open(files, plugin=plugin, stack=stack)
    return viewer


class BaseFrame(QtWidgets.QFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)

        self.figure = Figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvas(self.figure)
        self.orig_xlim = None
        self.orig_ylim = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return self.figure, self.axes, self.canvas

    def resetView(self):
        self.axes.set_xlim(self.orig_xlim)
        self.axes.set_ylim(self.orig_ylim)


class DragDropListView(QtWidgets.QListView):
    def __init__(self, parent=None):
        super(DragDropListView, self).__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self._default_style = self.styleSheet()

    @property
    def items(self):
        for i in range(self.model.rowCount()):
            yield self.model.item(i)

    @property
    def state(self):
        tests = [item.checkState() == QtCore.Qt.Checked for item in self.items]
        n_checked = sum(tests)
        if all(tests):
            state = QtCore.Qt.Checked
        elif any(tests):
            state = QtCore.Qt.PartiallyChecked
        else:
            state = QtCore.Qt.Unchecked
        return state, n_checked

    def add_item(self, path):
        item = QStandardItem(path)
        item.setCheckable(True)
        item.setCheckState(QtCore.Qt.Checked)
        self.model.appendRow(item)

    def clear(self):
        self.model.removeRows(0, self.model.rowCount())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.add_item(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if not file.startswith("."):
                            self.add_item(os.path.join(root, file))


class ItemSelectionFrame(QtWidgets.QFrame):
    def __init__(self, items, parent=None):
        super(ItemSelectionFrame, self).__init__(parent)
        self.setFrameShape(self.Shape.StyledPanel)
        self.setLineWidth(0)

        self.select_box = QtWidgets.QCheckBox("Files")
        self.select_box.setChecked(True)
        self.select_box.stateChanged.connect(self.toggle_select)

        self.fancy_list = DragDropListView(self)
        self._model = self.fancy_list.model
        self._model.rowsInserted.connect(self.check_select_box)
        self._model.rowsRemoved.connect(self.check_select_box)
        self._model.itemChanged.connect(self.check_select_box)
        for item in items:
            self.fancy_list.add_item(item)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.select_box)
        self.layout.addWidget(self.fancy_list)

    @property
    def selected_items(self):
        for item in self.fancy_list.items:
            if item.checkState() == QtCore.Qt.Checked:
                yield item.text()

    def check_select_box(self):
        state, n_checked = self.fancy_list.state
        if self.select_box.checkState() != state:
            self.select_box.blockSignals(True)
            self.select_box.setCheckState(state)
            self.select_box.blockSignals(False)
        string = "file"
        if n_checked > 1:
            string += "s"
        self.select_box.setText(f"{n_checked} {string} selected")

    def toggle_select(self, state):
        state = QtCore.Qt.CheckState(state)
        if state == QtCore.Qt.PartiallyChecked:
            return
        for item in self.fancy_list.items:
            if item.checkState() != state:
                item.setCheckState(state)


class NavigationToolbar(NavigationToolbar2QT):
    toolitems = [
        t for t in NavigationToolbar2QT.toolitems if t[0] in ("Home", "Pan", "Zoom")
    ]

    def set_message(self, msg):
        pass

    def release_zoom(self, event):
        super(NavigationToolbar, self).release_zoom(event)
        self.zoom()


class StreamWriter:
    def __init__(self):
        self.queue = Queue()

    def write(self, text):
        if text != "\n":
            self.queue.put(text)

    def flush(self):
        pass


class StreamReceiver(QtCore.QThread):
    new_text = QtCore.Signal(str)

    def __init__(self, queue):
        super(StreamReceiver, self).__init__()
        self.queue = queue

    def run(self):
        while True:
            text = self.queue.get()
            self.new_text.emit(text)


class ClickableLabel(QtWidgets.QLabel):
    signal = QtCore.Signal()

    def __init__(self, text="", color="turquoise", parent=None):
        super(ClickableLabel, self).__init__(text, parent)
        self._default_style = self.styleSheet()
        self.color = color
        self.setStyleSheet(f"color: {self.color}")

    def mouseReleaseEvent(self, event):
        self.signal.emit()

    def enterEvent(self, event):
        self.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.setStyleSheet(f"color: {self.color}")

    def leaveEvent(self, event):
        self.unsetCursor()
        self.setStyleSheet(self._default_style)


class ItemCreator(QtWidgets.QDialog):
    created = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    def __init__(self, parent=None):
        super(ItemCreator, self).__init__(parent)
        self.parent = parent
        vbox = QtWidgets.QVBoxLayout(self)
        self.field1 = QtWidgets.QLineEdit(self)
        self.field1.setPlaceholderText("Parameter")
        self.field2 = QtWidgets.QLineEdit(self)
        self.field2.setPlaceholderText("Value")
        create_button = QtWidgets.QPushButton(self)
        create_button.setText("Create")
        create_button.clicked.connect(self.form_item)
        vbox.addWidget(self.field1)
        vbox.addWidget(self.field2)
        vbox.addWidget(create_button)
        self.show()

    def form_item(self):
        key = self.field1.text()
        value = self.field2.text()
        item = QtWidgets.QTreeWidgetItem([key, value])
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        self.created.emit(item)
        self.accept()


# TODO Insert new video
# TODO Insert skeleton link
class ContextMenu(QtWidgets.QMenu):
    def __init__(self, parent):
        super(ContextMenu, self).__init__(parent)
        self.parent = parent
        self.current_item = parent.tree.currentItem()
        insert = QAction("Insert", self)
        insert.triggered.connect(self.create_item)
        delete = QAction("Delete", self)
        delete.triggered.connect(parent.remove_items)
        self.addAction(insert)
        self.addAction(delete)
        if self.current_item.text(0) == "project_path":
            fix_path = QAction("Fix Path", self)
            fix_path.triggered.connect(self.fix_path)
            self.addAction(fix_path)

    def create_item(self):
        creator = ItemCreator(self)
        creator.created.connect(self.parent.insert)

    def fix_path(self):
        self.current_item.setText(1, os.path.split(self.parent.filename)[0])


class CustomDelegate(QtWidgets.QItemDelegate):
    # Hack to make the first column read-only, as we do not want users to touch it.
    # The cleaner solution would be to use a QTreeView and QAbstractItemModel,
    # but that is a lot of rework for little benefits.
    def createEditor(self, parent, option, index):
        if index.column() != 0:
            return super(CustomDelegate, self).createEditor(parent, option, index)
        return None


class DictViewer(QtWidgets.QWidget):
    def __init__(self, cfg, filename="", parent=None):
        super(DictViewer, self).__init__(parent)
        self.cfg = cfg
        self.filename = filename
        self.parent = parent
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setItemDelegate(CustomDelegate())
        self.tree.setHeaderLabels(["Parameter", "Value"])
        self.tree.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)
        self.tree.itemChanged.connect(self.edit_value)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.pop_context_menu)

        self.root = self.tree.invisibleRootItem()
        self.tree.addTopLevelItem(self.root)
        self.populate_tree(cfg, self.root)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree)
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(QtWidgets.QLabel(filename))
        layout2.addWidget(self.tree)
        self.setLayout(layout2)

    def pop_context_menu(self, point):
        index = self.tree.indexAt(point)
        if not index.isValid():
            return
        menu = ContextMenu(self)
        menu.exec_(self.tree.mapToGlobal(point))

    def get_position_in_parent(self, item):
        parent = item.parent() or self.root
        index = parent.indexOfChild(item)
        return index, parent

    def insert(self, item):
        current = self.tree.selectedItems()[0]
        ind, parent = self.get_position_in_parent(current)
        parent.insertChild(ind + 1, item)

        value = self.cast_to_right_type(item.text(1))
        if parent is self.root:
            self.set_value(self.cfg, [item.text(0)], value)
        else:
            keys, _ = self.walk_recursively_to_root(current)
            self.set_value(self.cfg, keys, value, ind + 1)

    def remove(self, item):
        ind, parent = self.get_position_in_parent(item)
        keys, value = self.walk_recursively_to_root(item)
        if item.parent() and item.childCount():  # Handle nested dict or list
            keys = [keys[0], value]
        success = self.remove_key(self.cfg, keys, ind)
        if success:
            parent.removeChild(item)

    def remove_items(self):
        for item in self.tree.selectedItems():
            self.remove(item)

    @staticmethod
    def cast_to_right_type(val):
        try:
            val = ast.literal_eval(val)
        except ValueError:
            # Leave untouched when it is already a string
            pass
        except SyntaxError:
            # Slashes also raise the error, but no need to print anything since it is then likely to be a path
            if os.path.sep not in val:
                print("Consider removing leading zeros or spaces in the string.")
        return val

    @staticmethod
    def walk_recursively_to_root(item):
        vals = []
        # Walk backwards across parents to get all keys
        while item is not None:
            for i in range(item.columnCount() - 1, -1, -1):
                vals.append(item.text(i))
            item = item.parent()
        *keys, value = vals[::-1]
        return keys, value

    @staticmethod
    def get_nested_key(cfg, keys):
        temp = cfg
        for key in keys[:-1]:
            try:
                temp = temp.setdefault(key, {})
            except AttributeError:  # Handle nested lists
                temp = temp[int(key)]
        return temp

    def edit_value(self, item):
        keys, value = self.walk_recursively_to_root(item)
        if (
            "crop" not in keys
        ):  # 'crop' should not be cast, otherwise it is understood as a list
            value = self.cast_to_right_type(value)
        self.set_value(self.cfg, keys, value)

    def set_value(self, cfg, keys, value, ind=None):
        temp = self.get_nested_key(cfg, keys)
        try:  # Work for a dict
            temp[keys[-1]] = value
        except TypeError:  # Needed to index a list
            if ind is None:  # Edit the list in place
                temp[self.tree.currentIndex().row()] = value
            else:
                temp.insert(ind, value)

    def remove_key(self, cfg, keys, ind=None):
        if not len(keys):  # Avoid deleting a parent list or dict
            return
        temp = self.get_nested_key(cfg, keys)
        try:
            temp.pop(keys[-1])
        except TypeError:
            if ind is None:
                ind = self.tree.currentIndex().row()
            temp.pop(ind)
        return True

    def populate_tree(self, data, tree_widget):
        if isinstance(data, dict):
            for key, val in data.items():
                self.add_row(key, val, tree_widget)
        elif isinstance(data, list):
            for i, val in enumerate(data):
                self.add_row(str(i), val, tree_widget)
        else:
            print("This should never be reached!")

    def add_row(self, key, val, tree_widget):
        if isinstance(val, dict) or isinstance(val, list):
            item = QtWidgets.QTreeWidgetItem([key])
            self.populate_tree(val, item)
        else:
            item = QtWidgets.QTreeWidgetItem([key, str(val)])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        tree_widget.addChild(item)


class ConfigEditor(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super(ConfigEditor, self).__init__(parent)
        self.config = config
        if (
            config.endswith("config.yaml")
            and not config.endswith("pytorch_config.yaml")
        ):
            self.read_func = auxiliaryfunctions.read_config
            self.write_func = auxiliaryfunctions.write_config
        else:
            self.read_func = auxiliaryfunctions.read_plainconfig
            self.write_func = auxiliaryfunctions.write_plainconfig
        self.cfg = self.read_func(config)
        self.parent = parent
        self.setWindowTitle("Configuration Editor")
        if parent is not None:
            self.setMinimumWidth(parent.screen_width // 2)
            self.setMinimumHeight(parent.screen_height // 2)
        self.viewer = DictViewer(self.cfg, config, self)

        self.save_button = QtWidgets.QPushButton("Save", self)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.close)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.viewer)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.save_button)
        hbox.addWidget(self.cancel_button)
        vbox.addLayout(hbox)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def accept(self):
        self.write_func(self.config, self.cfg)
        super(ConfigEditor, self).accept()


class FrameCropper(QtWidgets.QDialog):
    def __init__(self, video, parent=None):
        super(FrameCropper, self).__init__(parent)
        self.clip = VideoWriter(video)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax_help = self.fig.add_axes([0.9, 0.2, 0.1, 0.1])
        self.ax_save = self.fig.add_axes([0.9, 0.1, 0.1, 0.1])
        self.crop_button = Button(self.ax_save, "Crop")
        self.crop_button.on_clicked(self.validate_crop)
        self.help_button = Button(self.ax_help, "Help")
        self.help_button.on_clicked(self.display_help)

        self.canvas = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.bbox = [0, 0, 0, 0]

    def draw_bbox(self):
        frame = None
        # Read the video until a frame is successfully read
        while frame is None:
            frame = self.clip.read_frame()
        self.bbox[-2:] = frame.shape[1], frame.shape[0]
        self.ax.imshow(frame[:, :, ::-1])

        self.rs = RectangleSelector(
            self.ax,
            self.line_select_callback,
            minspanx=5,
            minspany=5,
            interactive=True,
            spancoords="pixels",
        )
        self.show()
        self.fig.canvas.start_event_loop(timeout=-1)
        return self.bbox

    def line_select_callback(self, eclick, erelease):
        self.bbox[:2] = int(eclick.xdata), int(eclick.ydata)  # x1, y1
        self.bbox[2:] = int(erelease.xdata), int(erelease.ydata)  # x2, y2

    def validate_crop(self, *args):
        self.fig.canvas.stop_event_loop()
        self.close()

    def display_help(self, *args):
        print(
            "1. Use left click to select the region of interest. A red box will be drawn around the selected region. \n\n2. Use the corner points to expand the box and center to move the box around the image. \n\n3. Click "
        )


class SkeletonBuilder(QtWidgets.QDialog):
    def __init__(self, config_path, parent=None):
        super(SkeletonBuilder, self).__init__(parent)
        self.config_path = config_path
        self.cfg = auxiliaryfunctions.read_config(config_path)
        # Find uncropped labeled data
        self.df = None
        found = False
        root = os.path.join(self.cfg["project_path"], "labeled-data")
        for dir_ in os.listdir(root):
            folder = os.path.join(root, dir_)
            if os.path.isdir(folder) and not any(
                folder.endswith(s) for s in ("cropped", "labeled")
            ):
                self.df = pd.read_hdf(
                    os.path.join(folder, f'CollectedData_{self.cfg["scorer"]}.h5')
                )
                row, col = self.pick_labeled_frame()
                if "individuals" in self.df.columns.names:
                    self.df = self.df.xs(col, axis=1, level="individuals")
                self.xy = self.df.loc[row].values.reshape((-1, 2))
                missing = np.flatnonzero(np.isnan(self.xy).all(axis=1))
                if not missing.size:
                    found = True
                    break
        if self.df is None:
            raise IOError("No labeled data were found.")

        self.bpts = self.df.columns.get_level_values("bodyparts").unique()
        if not found:
            warnings.warn(
                f"A fully labeled animal could not be found. "
                f"{', '.join(self.bpts[missing])} will need to be manually connected in the config.yaml."
            )
        self.tree = KDTree(self.xy)
        # Handle image previously annotated on a different platform
        if isinstance(row, str):
            sep = "/" if "/" in row else "\\"
            row = row.split(sep)
        self.image = io.imread(os.path.join(self.cfg["project_path"], *row))
        self.inds = set()
        self.segs = set()
        # Draw the skeleton if already existent
        if self.cfg["skeleton"]:
            for bone in self.cfg["skeleton"]:
                pair = np.flatnonzero(self.bpts.isin(bone))
                if len(pair) != 2:
                    continue
                pair_sorted = tuple(sorted(pair))
                self.inds.add(pair_sorted)
                self.segs.add(tuple(map(tuple, self.xy[pair_sorted, :])))

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        ax_clear = self.fig.add_axes([0.85, 0.55, 0.1, 0.1])
        ax_export = self.fig.add_axes([0.85, 0.45, 0.1, 0.1])
        self.clear_button = Button(ax_clear, "Clear")
        self.clear_button.on_clicked(self.clear)
        self.export_button = Button(ax_export, "Export")
        self.export_button.on_clicked(self.export)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.lines = LineCollection(
            self.segs, colors=mcolors.to_rgba(self.cfg["skeleton_color"])
        )
        self.lines.set_picker(True)
        self._show()

    def pick_labeled_frame(self):
        # Find the most 'complete' animal
        try:
            count = self.df.groupby(level="individuals", axis=1).count()
            if "single" in count:
                count.drop("single", axis=1, inplace=True)
        except KeyError:
            count = self.df.count(axis=1).to_frame()
        mask = count.where(count == count.values.max())
        kept = mask.stack().index.to_list()
        np.random.shuffle(kept)
        picked = kept.pop()
        row = picked[:-1]
        col = picked[-1]
        return row, col

    def _show(self):
        lo = np.nanmin(self.xy, axis=0)
        hi = np.nanmax(self.xy, axis=0)
        center = (hi + lo) / 2
        w, h = hi - lo
        ampl = 1.3
        w *= ampl
        h *= ampl
        self.ax.set_xlim(center[0] - w / 2, center[0] + w / 2)
        self.ax.set_ylim(center[1] - h / 2, center[1] + h / 2)
        self.ax.imshow(self.image)
        self.ax.scatter(*self.xy.T, s=self.cfg["dotsize"] ** 2)
        self.ax.add_collection(self.lines)
        self.ax.invert_yaxis()

        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.show()

    def clear(self, *args):
        self.inds.clear()
        self.segs.clear()
        self.lines.set_segments(self.segs)

    def export(self, *args):
        inds_flat = set(ind for pair in self.inds for ind in pair)
        unconnected = [i for i in range(len(self.xy)) if i not in inds_flat]
        if len(unconnected):
            warnings.warn(
                f"You didn't connect all the bodyparts (which is fine!). This is just a note to let you know."
            )
        self.cfg["skeleton"] = [tuple(self.bpts[list(pair)]) for pair in self.inds]
        auxiliaryfunctions.write_config(self.config_path, self.cfg)

    def on_pick(self, event):
        if event.mouseevent.button == 3:
            removed = event.artist.get_segments().pop(event.ind[0])
            self.segs.remove(tuple(map(tuple, removed)))
            self.inds.remove(tuple(self.tree.query(removed)[1]))

    def on_select(self, verts):
        self.path = Path(verts)
        self.verts = verts
        inds = self.tree.query_ball_point(verts, 5)
        inds_unique = []
        for lst in inds:
            if len(lst) and lst[0] not in inds_unique:
                inds_unique.append(lst[0])
        for pair in zip(inds_unique, inds_unique[1:]):
            pair_sorted = tuple(sorted(pair))
            self.inds.add(pair_sorted)
            self.segs.add(tuple(map(tuple, self.xy[pair_sorted, :])))
        self.lines.set_segments(self.segs)
        self.fig.canvas.draw_idle()
