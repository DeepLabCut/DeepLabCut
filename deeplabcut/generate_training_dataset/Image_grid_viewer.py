from PySide6 import QtWidgets, QtCore, QtGui
import os
from pathlib import Path
from PIL import Image, ImageQt

class QtImageGridViewer(QtWidgets.QDialog):
    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self.setWindowTitle("DLC Frames Cleanup Tool")
        self.resize(1200, 800)

        # Initialize variables
        self.grid_size = (5, 5)
        self.fixed_width = 160
        self.fixed_height = 120
        self.border_size = 3
        self.image_files = []
        self.current_page = 0
        self.images_per_page = self.grid_size[0] * self.grid_size[1]
        self.selected_images = set()
        self.last_selected = None
        self.config_path = config_path

        # Get project colors
        if parent:
            self.theme_color = parent.palette().color(QtGui.QPalette.Highlight)
            self.text_color = parent.palette().color(QtGui.QPalette.Text)
        else:
            self.theme_color = QtGui.QColor(42, 130, 218)  # DLC blue
            self.text_color = QtGui.QColor(0, 0, 0)

        # Create main layout
        self.main_layout = QtWidgets.QVBoxLayout(self)

        # Create menu bar
        self.create_menu()

        # Create scroll area for images
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)

        # Create grid container widget
        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(5)
        self.scroll_area.setWidget(self.grid_widget)

        # Create selection controls
        self.create_selection_controls()

        # Create navigation controls
        self.create_navigation_controls()

        # If config path provided, try to load images
        if config_path:
            self.auto_load_images()

    def create_menu(self):
        """Create the menu bar with options"""
        self.menu_bar = QtWidgets.QMenuBar()
        self.main_layout.setMenuBar(self.menu_bar)

        # File menu
        file_menu = self.menu_bar.addMenu("File")
        load_action = QtGui.QAction("Load Images", self)
        load_action.triggered.connect(self.load_images)
        file_menu.addAction(load_action)

        # Settings menu
        settings_menu = self.menu_bar.addMenu("Settings")
        grid_action = QtGui.QAction("Set Grid Size", self)
        grid_action.triggered.connect(self.set_grid_size)
        settings_menu.addAction(grid_action)

        image_action = QtGui.QAction("Set Image Size", self)
        image_action.triggered.connect(self.set_image_size)
        settings_menu.addAction(image_action)

    def create_selection_controls(self):
        """Create buttons for selection management"""
        selection_frame = QtWidgets.QFrame()
        selection_layout = QtWidgets.QHBoxLayout(selection_frame)

        # Selection buttons
        self.select_all_btn = QtWidgets.QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        selection_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QtWidgets.QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        selection_layout.addWidget(self.deselect_all_btn)

        # Help label
        help_label = QtWidgets.QLabel("Shift+Click: Select Range | Ctrl+Click: Add to Selection")
        help_label.setStyleSheet("color: blue; font-size: 8pt;")
        selection_layout.addWidget(help_label, 1)

        # Count and delete
        self.selection_count = QtWidgets.QLabel("Selected: 0")
        selection_layout.addWidget(self.selection_count)

        self.delete_btn = QtWidgets.QPushButton("Delete Selected")
        self.delete_btn.setStyleSheet("background-color: red; color: white;")
        self.delete_btn.clicked.connect(self.delete_selected)
        selection_layout.addWidget(self.delete_btn)

        self.main_layout.addWidget(selection_frame)

    def create_navigation_controls(self):
        """Create navigation buttons for paging"""
        nav_frame = QtWidgets.QFrame()
        nav_layout = QtWidgets.QHBoxLayout(nav_frame)

        self.prev_btn = QtWidgets.QPushButton("Previous Page")
        self.prev_btn.clicked.connect(self.prev_page)
        nav_layout.addWidget(self.prev_btn)

        self.page_label = QtWidgets.QLabel("Page: 1")
        nav_layout.addWidget(self.page_label)

        self.next_btn = QtWidgets.QPushButton("Next Page")
        self.next_btn.clicked.connect(self.next_page)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addStretch(1)

        self.main_layout.addWidget(nav_frame)

    def auto_load_images(self):
        """Attempt to auto-load images from labeled-data directory"""
        if self.config_path:
            config_dir = Path(self.config_path).parent
            labeled_data_dir = config_dir / "labeled-data"

            if labeled_data_dir.exists() and labeled_data_dir.is_dir():
                # Find all subdirectories (dataset folders)
                datasets = [d for d in labeled_data_dir.iterdir() if d.is_dir()]

                if datasets:
                    # Let user choose which dataset to load
                    dialog = QtWidgets.QDialog(self)
                    dialog.setWindowTitle("Select Dataset")
                    dialog_layout = QtWidgets.QVBoxLayout(dialog)

                    label = QtWidgets.QLabel("Select which dataset to clean up:")
                    dialog_layout.addWidget(label)

                    list_widget = QtWidgets.QListWidget()
                    for dataset in datasets:
                        list_widget.addItem(dataset.name)
                    dialog_layout.addWidget(list_widget)

                    buttons = QtWidgets.QDialogButtonBox(
                        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
                    )
                    buttons.accepted.connect(dialog.accept)
                    buttons.rejected.connect(dialog.reject)
                    dialog_layout.addWidget(buttons)

                    if dialog.exec_() == QtWidgets.QDialog.Accepted and list_widget.currentItem():
                        dataset_name = list_widget.currentItem().text()
                        self.load_directory(str(labeled_data_dir / dataset_name))

    def load_images(self):
        """Open a file dialog to select the directory with images"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )

        if directory:
            self.load_directory(directory)

    def load_directory(self, directory):
        """Load all images from the specified directory"""
        self.image_files = []
        for file in os.listdir(directory):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_files.append(os.path.join(directory, file))

        # Sort alphabetically by default (instead of random)
        self.image_files.sort()

        self.current_page = 0
        self.images_per_page = self.grid_size[0] * self.grid_size[1]
        self.selected_images = set()
        self.last_selected = None
        self.update_selection_count()
        self.display_grid()

    def set_grid_size(self):
        """Open dialog to set grid dimensions"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set Grid Size")
        layout = QtWidgets.QFormLayout(dialog)

        row_spin = QtWidgets.QSpinBox()
        row_spin.setRange(1, 20)
        row_spin.setValue(self.grid_size[0])
        layout.addRow("Rows:", row_spin)

        col_spin = QtWidgets.QSpinBox()
        col_spin.setRange(1, 20)
        col_spin.setValue(self.grid_size[1])
        layout.addRow("Columns:", col_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            old_images_per_page = self.images_per_page
            first_visible_image = self.current_page * old_images_per_page

            self.grid_size = (row_spin.value(), col_spin.value())
            self.images_per_page = self.grid_size[0] * self.grid_size[1]

            # Adjust current page
            if self.image_files and old_images_per_page != 0:
                self.current_page = first_visible_image // self.images_per_page

            # Verify current page is valid
            if self.image_files:
                max_page = max(0, (len(self.image_files) - 1) // self.images_per_page)
                self.current_page = min(self.current_page, max_page)

            self.display_grid()

    def set_image_size(self):
        """Open dialog to set image thumbnail size"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Set Image Size")
        layout = QtWidgets.QFormLayout(dialog)

        width_spin = QtWidgets.QSpinBox()
        width_spin.setRange(50, 800)
        width_spin.setValue(self.fixed_width)
        layout.addRow("Width:", width_spin)

        height_spin = QtWidgets.QSpinBox()
        height_spin.setRange(50, 800)
        height_spin.setValue(self.fixed_height)
        layout.addRow("Height:", height_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.fixed_width = width_spin.value()
            self.fixed_height = height_spin.value()
            self.display_grid()

    def display_grid(self):
        """Display the current page of images in a grid"""
        # Clear all widgets from grid
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not self.image_files:
            return

        # Calculate start index for current page
        start_idx = self.current_page * self.images_per_page

        # Update page label
        total_pages = max(1, (len(self.image_files) + self.images_per_page - 1) // self.images_per_page)
        self.page_label.setText(f"Page: {self.current_page + 1} of {total_pages}")

        # Update prev/next buttons
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled((self.current_page + 1) * self.images_per_page < len(self.image_files))

        # Create image cells
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                img_idx = start_idx + row * self.grid_size[1] + col

                if img_idx < len(self.image_files):
                    try:
                        # Create cell container
                        cell_widget = QtWidgets.QWidget()
                        cell_layout = QtWidgets.QVBoxLayout(cell_widget)
                        cell_layout.setContentsMargins(3, 3, 3, 3)

                        # Create checkbox
                        checkbox = QtWidgets.QCheckBox()
                        checkbox.setChecked(img_idx in self.selected_images)
                        checkbox.stateChanged.connect(lambda state, idx=img_idx: self.toggle_selection(idx, state))
                        cell_layout.addWidget(checkbox, 0, QtCore.Qt.AlignLeft)

                        # Load and resize image
                        img = Image.open(self.image_files[img_idx])
                        img = img.resize((self.fixed_width, self.fixed_height))
                        qimage = ImageQt.ImageQt(img)
                        pixmap = QtGui.QPixmap.fromImage(qimage)

                        # Create image label with border
                        img_frame = QtWidgets.QFrame()
                        img_frame.setFrameShape(QtWidgets.QFrame.Box)
                        img_frame.setLineWidth(self.border_size)
                        img_frame.setStyleSheet(
                            f"border: {self.border_size}px solid {'red' if img_idx in self.selected_images else 'gray'};"
                        )

                        img_layout = QtWidgets.QVBoxLayout(img_frame)
                        img_layout.setContentsMargins(0, 0, 0, 0)

                        img_label = QtWidgets.QLabel()
                        img_label.setPixmap(pixmap)
                        img_label.setCursor(QtCore.Qt.PointingHandCursor)
                        img_label.mousePressEvent = lambda event, idx=img_idx: self.handle_image_click(event, idx)
                        img_layout.addWidget(img_label)

                        cell_layout.addWidget(img_frame)

                        # Add filename label
                        filename = os.path.basename(self.image_files[img_idx])
                        short_name = filename[:15] + "..." if len(filename) > 15 else filename
                        name_label = QtWidgets.QLabel(short_name)
                        name_label.setAlignment(QtCore.Qt.AlignCenter)
                        name_label.setToolTip(filename)
                        name_label.setStyleSheet("font-size: 8pt;")
                        cell_layout.addWidget(name_label)

                        self.grid_layout.addWidget(cell_widget, row, col)

                    except Exception as e:
                        # Error placeholder
                        error_label = QtWidgets.QLabel(f"Error: {str(e)}")
                        error_label.setStyleSheet("background-color: gray;")
                        self.grid_layout.addWidget(error_label, row, col)

    def handle_image_click(self, event, img_idx):
        """Handle clicks on images with modifier key support"""
        ctrl_pressed = event.modifiers() & QtCore.Qt.ControlModifier
        shift_pressed = event.modifiers() & QtCore.Qt.ShiftModifier

        if shift_pressed and self.last_selected is not None:
            # Shift-click: select range
            start_idx = min(self.last_selected, img_idx)
            end_idx = max(self.last_selected, img_idx)

            # Get all visible indices in range
            start_page_idx = self.current_page * self.images_per_page
            end_page_idx = min(start_page_idx + self.images_per_page, len(self.image_files))
            visible_indices = list(range(start_page_idx, end_page_idx))

            # Select all visible indices in range
            for idx in visible_indices:
                if start_idx <= idx <= end_idx:
                    self.selected_images.add(idx)

            self.last_selected = img_idx
            self.update_selection_count()
            self.display_grid()
            return

        if not ctrl_pressed:
            # If not Ctrl, deselect others
            if img_idx not in self.selected_images:
                self.selected_images.clear()

        # Toggle current image
        if img_idx in self.selected_images:
            self.selected_images.discard(img_idx)
        else:
            self.selected_images.add(img_idx)

        self.last_selected = img_idx
        self.update_selection_count()
        self.display_grid()

    def toggle_selection(self, img_idx, state):
        """Handle checkbox state changes"""
        if state == QtCore.Qt.Checked:
            self.selected_images.add(img_idx)
            self.last_selected = img_idx
        else:
            self.selected_images.discard(img_idx)

        self.update_selection_count()
        self.display_grid()

    def update_selection_count(self):
        """Update the selection counter label"""
        self.selection_count.setText(f"Selected: {len(self.selected_images)}")

    def select_all(self):
        """Select all images on current page"""
        start_idx = self.current_page * self.images_per_page
        end_idx = min(start_idx + self.images_per_page, len(self.image_files))

        for i in range(start_idx, end_idx):
            self.selected_images.add(i)

        self.update_selection_count()
        self.display_grid()

    def deselect_all(self):
        """Deselect all images"""
        self.selected_images.clear()
        self.update_selection_count()
        self.display_grid()

    def delete_selected(self):
        """Delete selected images"""
        if not self.selected_images:
            QtWidgets.QMessageBox.information(
                self, "No Selection", "No images selected for deletion."
            )
            return

        # Confirm deletion
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(self.selected_images)} selected images?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if confirm != QtWidgets.QMessageBox.Yes:
            return

        # Get paths to delete
        files_to_delete = [self.image_files[i] for i in self.selected_images]

        # Remove from list
        self.image_files = [f for i, f in enumerate(self.image_files) if i not in self.selected_images]

        # Ask about physical deletion
        delete_from_disk = QtWidgets.QMessageBox.question(
            self,
            "Delete from Disk",
            "Do you also want to delete these files from disk? (This cannot be undone)",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if delete_from_disk == QtWidgets.QMessageBox.Yes:
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                except Exception as e:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Deletion Error",
                        f"Could not delete {os.path.basename(file_path)}: {str(e)}"
                    )

        # Clear selection and adjust page
        self.selected_images.clear()
        self.last_selected = None
        self.update_selection_count()

        total_pages = max(1, (len(self.image_files) + self.images_per_page - 1) // self.images_per_page)
        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)

        self.display_grid()

        # Success message
        QtWidgets.QMessageBox.information(
            self,
            "Deletion Complete",
            f"{len(files_to_delete)} images removed from the viewer."
            + (f" Files were also deleted from disk." if delete_from_disk == QtWidgets.QMessageBox.Yes else "")
        )

    def prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_grid()

    def next_page(self):
        """Go to next page"""
        if (self.current_page + 1) * self.images_per_page < len(self.image_files):
            self.current_page += 1
            self.display_grid()

    def on_resize(self, event):
        # Update scrollbar region when window is resized
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.updateGeometry()
        self.scroll_area.setMinimumSize(self.sizeHint())
        self.scroll_area.setMaximumSize(self.sizeHint())

    def show_tooltip(self, event, text):
        # Use PySide6's QToolTip static method to show tooltip
        position = event.globalPos()
        QtWidgets.QToolTip.showText(position, text)

    def hide_tooltip(self):
        # Hide any active tooltip
        QtWidgets.QToolTip.hideText()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    viewer = QtImageGridViewer()
    viewer.resize(1080, 900)
    viewer.show()
    sys.exit(app.exec())
