import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QSlider,
    QHBoxLayout, QWidget, QMessageBox, QDialog, QCheckBox, QScrollArea, QGridLayout,
    QFrame, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QIntValidator
from PyQt5.QtCore import Qt

class ColorSwatch(QFrame):
    """A custom widget that displays a colored square representing a cluster's color"""
    def __init__(self, color, size=20):
        super().__init__()
        self.setFixedSize(size, size)
        self.color = color
        # Create a styled square with the cluster's color
        self.setStyleSheet(
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); "
            f"border: 1px solid black;"
        )

class ClusterInfo(QWidget):
    """Widget showing information about a specific cluster, including its color and index"""
    def __init__(self, cluster_idx, color, size=20):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add color swatch
        swatch = ColorSwatch(color, size)
        layout.addWidget(swatch)
        
        # Add cluster information label
        info_label = QLabel(f"Cluster {cluster_idx + 1}: RGB({color[0]}, {color[1]}, {color[2]})")
        layout.addWidget(info_label)
        layout.addStretch()
        
        self.setLayout(layout)

class ImageFilterDialog(QDialog):
    """Dialog for applying various image filters before segmentation with smooth transitions"""
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Filters")
        self.setGeometry(300, 300, 800, 600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.original_image = image.copy()
        self.filtered_image = image.copy()
        self.current_params = {}  # Store current parameter values
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Create filter controls section with improved styling
        controls_group = QGroupBox("Filter Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 1ex;
                padding: 10px;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        controls_layout = QVBoxLayout()
        
        # Filter type selection dropdown with improved styling
        filter_row = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                color: #333333;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #4a90e2;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 5px;
            }
        """)
        self.filter_combo.addItems([
            "No Filter",
            "Gaussian Blur",
            "Median Blur",
            "Bilateral Filter",
            "Sharpen"
        ])
        self.filter_combo.currentTextChanged.connect(self.update_filter_controls)
        filter_row.addWidget(QLabel("Filter Type:"))
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        controls_layout.addLayout(filter_row)
        
        # Container for filter-specific parameters
        self.params_widget = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_widget.setLayout(self.params_layout)
        controls_layout.addWidget(self.params_widget)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Create preview section with improved styling
        preview_group = QGroupBox("Preview")
        preview_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 1ex;
                padding: 10px;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        preview_layout = QHBoxLayout()
        
        # Original image preview
        original_container = QWidget()
        original_layout = QVBoxLayout()
        original_label = QLabel("Original")
        original_label.setAlignment(Qt.AlignCenter)
        self.original_preview = QLabel()
        self.original_preview.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_label)
        original_layout.addWidget(self.original_preview)
        original_container.setLayout(original_layout)
        
        # Filtered image preview
        filtered_container = QWidget()
        filtered_layout = QVBoxLayout()
        filtered_label = QLabel("Filtered")
        filtered_label.setAlignment(Qt.AlignCenter)
        self.filtered_preview = QLabel()
        self.filtered_preview.setAlignment(Qt.AlignCenter)
        filtered_layout.addWidget(filtered_label)
        filtered_layout.addWidget(self.filtered_preview)
        filtered_container.setLayout(filtered_layout)
        
        preview_layout.addWidget(original_container)
        preview_layout.addWidget(filtered_container)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Add button row with improved styling
        button_layout = QHBoxLayout()
        button_style = """
            QPushButton {
                padding: 8px 16px;
                border: 2px solid #4a90e2;
                border-radius: 4px;
                background-color: white;
                color: #4a90e2;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a90e2;
                color: white;
            }
            QPushButton:pressed {
                background-color: #357abd;
                border-color: #357abd;
            }
        """
        
        reset_button = QPushButton("Reset")
        apply_button = QPushButton("Apply")
        cancel_button = QPushButton("Cancel")
        
        for button in [reset_button, apply_button, cancel_button]:
            button.setStyleSheet(button_style)
        
        reset_button.clicked.connect(self.reset_filters)
        apply_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Initialize the preview images
        self.update_previews()

    def update_parameter(self, param_name, value):
        """Updates a filter parameter and reapplies the filter"""
        self.current_params[param_name] = value
        self.apply_filter()

    def create_parameter_widget(self, param_name, min_val, max_val, default_val, step=1, is_float=False):
        """Creates a parameter control widget with native Qt arrows"""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f"{param_name}:"))
        
        # Simplified spin box styling using native arrows
        spin_style = """
            QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                color: #333333;
                min-width: 80px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #4a90e2;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                min-width: 16px;
                min-height: 12px;
                margin-top: 1px;
                margin-right: 1px;
                background: #f5f5f5;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                min-width: 16px;
                min-height: 12px;
                margin-bottom: 1px;
                margin-right: 1px;
                background: #f5f5f5;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #e0e0e0;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background: #d0d0d0;
            }
        """
        
        if is_float:
            spin = QDoubleSpinBox()
            spin.setDecimals(1)
        else:
            spin = QSpinBox()
        
        spin.setRange(min_val, max_val)
        spin.setValue(default_val)
        spin.setSingleStep(step)
        spin.setStyleSheet(spin_style)
        
        # Store the current value
        self.current_params[param_name] = default_val
        spin.valueChanged.connect(lambda value: self.update_parameter(param_name, value))
        
        layout.addWidget(spin)
        layout.addStretch()
        return layout

    def update_filter_controls(self):
        """Updates the parameter controls based on the selected filter type"""
        # Clear existing parameter widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
    
        filter_type = self.filter_combo.currentText()
        
        if filter_type == "Gaussian Blur":
            self.params_layout.addLayout(
                self.create_parameter_widget("Kernel Size", 1, 31, 
                    self.current_params.get("Kernel Size", 5))
            )
            self.params_layout.addLayout(
                self.create_parameter_widget("Sigma", 0.1, 5.0, 
                    self.current_params.get("Sigma", 1.0), 0.1, True)
            )
            
        elif filter_type == "Median Blur":
            self.params_layout.addLayout(
                self.create_parameter_widget("Kernel Size", 1, 31,
                    self.current_params.get("Kernel Size", 5))
            )
            
        elif filter_type == "Bilateral Filter":
            self.params_layout.addLayout(
                self.create_parameter_widget("Diameter", 1, 31,
                    self.current_params.get("Diameter", 9))
            )
            self.params_layout.addLayout(
                self.create_parameter_widget("Sigma Color", 1, 150,
                    self.current_params.get("Sigma Color", 75))
            )
            self.params_layout.addLayout(
                self.create_parameter_widget("Sigma Space", 1, 150,
                    self.current_params.get("Sigma Space", 75))
            )
            
        elif filter_type == "Sharpen":
            self.params_layout.addLayout(
                self.create_parameter_widget("Amount", 0.1, 5.0,
                    self.current_params.get("Amount", 1.5), 0.1, True)
            )
        
        self.apply_filter()
        
    def apply_filter(self):
        """Applies the selected filter with current parameters to the image"""
        filter_type = self.filter_combo.currentText()
        
        if filter_type == "No Filter":
            self.filtered_image = self.original_image.copy()
            
        elif filter_type == "Gaussian Blur":
            kernel_size = self.current_params.get("Kernel Size", 5)
            sigma = self.current_params.get("Sigma", 1.0)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            self.filtered_image = cv2.GaussianBlur(
                self.original_image,
                (kernel_size, kernel_size),
                sigma
            )
            
        elif filter_type == "Median Blur":
            kernel_size = self.current_params.get("Kernel Size", 5)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            self.filtered_image = cv2.medianBlur(
                self.original_image,
                kernel_size
            )
            
        elif filter_type == "Bilateral Filter":
            diameter = self.current_params.get("Diameter", 9)
            sigma_color = self.current_params.get("Sigma Color", 75)
            sigma_space = self.current_params.get("Sigma Space", 75)
            self.filtered_image = cv2.bilateralFilter(
                self.original_image,
                diameter,
                sigma_color,
                sigma_space
            )
            
        elif filter_type == "Sharpen":
            amount = self.current_params.get("Amount", 1.5)
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]) * amount
            self.filtered_image = cv2.filter2D(
                self.original_image,
                -1,
                kernel
            )
        
        self.update_previews()

    def update_previews(self):
        """Updates both preview images with current versions"""
        # Scale factor for preview images (30% of original size)
        scale_factor = 0.3
        
        # Update original preview
        preview_original = cv2.resize(
            self.original_image,
            None,
            fx=scale_factor,
            fy=scale_factor
        )
        self.display_preview(preview_original, self.original_preview)
        
        # Update filtered preview
        preview_filtered = cv2.resize(
            self.filtered_image,
            None,
            fx=scale_factor,
            fy=scale_factor
        )
        self.display_preview(preview_filtered, self.filtered_preview)

    def display_preview(self, image, label):
        """Displays an image in the specified label"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        label.setPixmap(QPixmap.fromImage(q_image))

    def reset_filters(self):
        """Resets all filters to their default state"""
        self.filter_combo.setCurrentText("No Filter")
        self.filtered_image = self.original_image.copy()
        self.update_previews()

    def get_filtered_image(self):
        """Returns the current filtered image"""
        return self.filtered_image

class UniformColorMaskDialog(QDialog):
    """Dialog for viewing and managing cluster masks"""
    def __init__(self, masks, dominant_colors, original_image, parent=None, initial_visibility=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Masks and Visibility")
        self.setGeometry(400, 300, 900, 700)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.parent = parent
        self.masks = masks
        self.colors = dominant_colors
        self.original_image = original_image
        self.cluster_visibility = initial_visibility or [True] * len(masks)
        self.checkboxes = []

        layout = QVBoxLayout()
        
        # Create grid layout for masks
        grid = QGridLayout()
        cols = 3  # Number of columns in the grid
        
        for idx, (mask, color) in enumerate(zip(masks, dominant_colors)):
            # Create container for each mask
            container = QWidget()
            container_layout = QVBoxLayout()
            
            # Add cluster info and checkbox
            info_container = QWidget()
            info_layout = QHBoxLayout()
            info_layout.setContentsMargins(0, 0, 0, 0)
            
            checkbox = QCheckBox()
            checkbox.setChecked(self.cluster_visibility[idx])
            checkbox.stateChanged.connect(lambda state, i=idx: self.update_cluster_visibility(i, state))
            info_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)
            
            cluster_info = ClusterInfo(idx, color)
            info_layout.addWidget(cluster_info)
            info_layout.addStretch()
            
            info_container.setLayout(info_layout)
            container_layout.addWidget(info_container)
            
            # Create and add mask image
            mask_color_image = np.zeros_like(original_image)
            mask_color_image[mask == 255] = color
            
            rgb_mask = cv2.cvtColor(mask_color_image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_mask.shape
            bytes_per_line = channel * width
            q_image = QImage(rgb_mask.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            mask_label = QLabel()
            mask_label.setPixmap(pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            mask_label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(mask_label)
            
            container.setLayout(container_layout)
            grid.addWidget(container, idx // cols, idx % cols)

        # Add grid to scroll area for better handling of many clusters
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_widget.setLayout(grid)
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Add preview section
        preview_group = QWidget()
        preview_layout = QVBoxLayout()
        preview_label = QLabel("Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        self.preview_image = QLabel()
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumHeight(250)
        preview_layout.addWidget(preview_label)
        preview_layout.addWidget(self.preview_image)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Add button row
        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset All")
        reset_button.clicked.connect(self.reset_visibility)
        apply_button = QPushButton("Apply Changes")
        apply_button.clicked.connect(self.apply_visibility)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        button_layout.addWidget(reset_button)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self.update_visibility()

    def update_cluster_visibility(self, idx, state):
        """Updates visibility state for a specific cluster"""
        self.cluster_visibility[idx] = state == Qt.Checked
        self.update_visibility()

    def update_visibility(self):
        """Updates the preview image based on current visibility settings"""
        combined_image = np.zeros_like(self.original_image)
        for idx, (mask, is_visible) in enumerate(zip(self.masks, self.cluster_visibility)):
            if is_visible:
                combined_image[mask == 255] = self.original_image[mask == 255]

        self.display_preview(combined_image)

    def display_preview(self, image):
        """Displays the preview image with proper scaling"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.preview_image.setPixmap(pixmap.scaled(
            self.preview_image.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def reset_visibility(self):
        """Resets all clusters to visible state"""
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)
        self.cluster_visibility = [True] * len(self.masks)
        self.update_visibility()

    def apply_visibility(self):
        """Applies current visibility settings to main window"""
        combined_image = np.zeros_like(self.original_image)
        for idx, (mask, is_visible) in enumerate(zip(self.masks, self.cluster_visibility)):
            if is_visible:
                combined_image[mask == 255] = self.original_image[mask == 255]

        self.parent.segmented_image = combined_image
        self.parent.cluster_visibility = self.cluster_visibility.copy()
        self.parent.display_image(combined_image)
        self.close()
        
class KMeansSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("K-means Image Segmentation")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create image display area with improved styling
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 500)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                color: #333333;
            }
        """)
        main_layout.addWidget(self.image_label)

        # Create controls section with improved styling
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Button row with improved styling
        button_layout = QHBoxLayout()
        
        # Enhanced button style
        button_style = """
            QPushButton {
                padding: 10px 20px;
                border: 2px solid #4a90e2;
                border-radius: 6px;
                background-color: #ffffff;
                color: #4a90e2;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4a90e2;
                color: white;
            }
            QPushButton:pressed {
                background-color: #357abd;
                border-color: #357abd;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                border-color: #cccccc;
                color: #999999;
            }
        """
        
        self.load_button = QPushButton("Load Image")
        self.filter_button = QPushButton("Apply Filters")
        self.segment_button = QPushButton("Segment Image")
        self.save_button = QPushButton("Save Segments")
        self.show_masks_button = QPushButton("Show Cluster Masks")
        self.help_button = QPushButton("Open Help PDF")  # New help button

        buttons = [
            self.load_button, self.filter_button, self.segment_button,
            self.save_button, self.show_masks_button, self.help_button  # Added help button to list
        ]
        
        for button in buttons:
            button.setStyleSheet(button_style)
            button_layout.addWidget(button)

        controls_layout.addLayout(button_layout)
        
        # Enhanced cluster control section
        cluster_group = QGroupBox("Clustering Settings")
        cluster_group.setStyleSheet("""
            QGroupBox {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                margin-top: 1ex;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        cluster_layout = QVBoxLayout(cluster_group)
        
        # Add manual cluster input
        manual_input_layout = QHBoxLayout()
        manual_input_label = QLabel("Number of Clusters:")
        manual_input_label.setStyleSheet("color: #333333;")
        self.cluster_input = QLineEdit()
        self.cluster_input.setValidator(QIntValidator(2, 10))
        self.cluster_input.setText("3")
        self.cluster_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 2px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                color: #333333;
            }
            QLineEdit:focus {
                border-color: #4a90e2;
            }
        """)
        self.cluster_input.setFixedWidth(60)
        manual_input_layout.addWidget(manual_input_label)
        manual_input_layout.addWidget(self.cluster_input)
        manual_input_layout.addStretch()
        cluster_layout.addLayout(manual_input_layout)
        
        # Enhanced slider
        slider_layout = QHBoxLayout()
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(2)
        self.k_slider.setMaximum(10)
        self.k_slider.setValue(3)
        self.k_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #e0e0e0;
                height: 8px;
                background: #f5f5f5;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 2px solid #4a90e2;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #357abd;
                border-color: #357abd;
            }
        """)
        
        slider_layout.addWidget(self.k_slider)
        cluster_layout.addLayout(slider_layout)
        
        controls_layout.addWidget(cluster_group)
        main_layout.addWidget(controls_widget)

        # Initialize button states and connections
        self.filter_button.setEnabled(False)
        self.show_masks_button.setEnabled(False)
        self.segment_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # Connect signals
        self.load_button.clicked.connect(self.load_image)
        self.filter_button.clicked.connect(self.show_filters)
        self.segment_button.clicked.connect(self.segment_image)
        self.save_button.clicked.connect(self.save_segments)
        self.show_masks_button.clicked.connect(self.show_masks)
        self.k_slider.valueChanged.connect(self.update_cluster_input)
        self.cluster_input.textChanged.connect(self.update_slider)

        # Initialize image-related variables
        self.original_image = None
        self.segmented_image = None
        self.dominant_colors = []
        self.masks = []
        self.cluster_visibility = []

        # Set light theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QWidget {
                background-color: white;
                color: #333333;
            }
            QLabel {
                color: #333333;
            }
        """)
        
        # Connect the new help button
        self.help_button.clicked.connect(self.open_help_pdf)
        
    def open_help_pdf(self):
        """Opens a PDF file with help documentation, accounting for both script and exe contexts"""
        try:
            # Get the application directory whether running as script or exe
            if getattr(sys, 'frozen', False):
                # Running as exe
                app_dir = os.path.dirname(sys.executable)
            else:
                # Running as script
                app_dir = os.path.dirname(os.path.abspath(__file__))
                
            default_manual = os.path.join(app_dir, "manual.pdf")
            
            if os.path.exists(default_manual):
                try:
                    # First try using default system associations
                    if sys.platform.startswith('darwin'):  # macOS
                        os.system(f'open "{default_manual}"')
                    elif sys.platform.startswith('win32'):  # Windows
                        os.startfile(default_manual)
                    else:  # Linux and other Unix-like
                        os.system(f'xdg-open "{default_manual}"')
                except Exception:
                    # If system association fails, try alternate methods
                    try:
                        import webbrowser
                        webbrowser.open(default_manual)
                    except Exception as e:
                        QMessageBox.critical(
                            self,
                            "Error",
                            f"Could not open PDF with system viewer: {str(e)}\n"
                            "Please ensure you have a PDF viewer installed."
                        )
            else:
                # Manual not found in application directory
                QMessageBox.warning(
                    self,
                    "Manual Not Found",
                    f"Could not find manual.pdf in:\n{app_dir}\n\n"
                    "Please ensure manual.pdf is in the same directory as the application."
                )
                
                # Optionally show file dialog to locate manual
                response = QMessageBox.question(
                    self,
                    "Locate Manual",
                    "Would you like to locate the manual PDF file?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if response == QMessageBox.Yes:
                    file_path, _ = QFileDialog.getOpenFileName(
                        self,
                        "Open Help PDF",
                        app_dir,  # Start in app directory
                        "PDF Files (*.pdf)"
                    )
                    if file_path:
                        # Try to copy the manual to the application directory
                        try:
                            import shutil
                            shutil.copy2(file_path, default_manual)
                            QMessageBox.information(
                                self,
                                "Manual Copied",
                                "The manual has been copied to the application directory for future use."
                            )
                            # Open the copied manual
                            if sys.platform.startswith('darwin'):
                                os.system(f'open "{default_manual}"')
                            elif sys.platform.startswith('win32'):
                                os.startfile(default_manual)
                            else:
                                os.system(f'xdg-open "{default_manual}"')
                        except Exception as e:
                            # If copy fails, just open the selected file
                            if sys.platform.startswith('darwin'):
                                os.system(f'open "{file_path}"')
                            elif sys.platform.startswith('win32'):
                                os.startfile(file_path)
                            else:
                                os.system(f'xdg-open "{file_path}"')
                                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to handle PDF: {str(e)}"
            )
            
    def update_cluster_input(self, value):
        """Updates the manual input when slider changes"""
        self.cluster_input.setText(str(value))

    def update_slider(self, text):
        """Updates the slider when manual input changes"""
        if text and text.isdigit():
            value = int(text)
            if 2 <= value <= 10:
                self.k_slider.setValue(value)

    def load_image(self):
        """Handles image loading with error checking and user feedback"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image File",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            try:
                # Load and validate image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Unable to load image")
                
                # Check image dimensions
                height, width = self.original_image.shape[:2]
                if height * width > 4000 * 3000:  # Limit image size to prevent performance issues
                    # Resize while maintaining aspect ratio
                    scale = min(4000/width, 3000/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    self.original_image = cv2.resize(
                        self.original_image,
                        (new_width, new_height),
                        interpolation=cv2.INTER_AREA
                    )
                    QMessageBox.information(
                        self,
                        "Image Resized",
                        "Image has been resized to improve performance while maintaining quality."
                    )
                
                self.display_image(self.original_image)
                self.filter_button.setEnabled(True)
                self.segment_button.setEnabled(True)
                self.setWindowTitle(f"K-means Image Segmentation - {file_path}")
                
                # Reset previous segmentation results
                self.segmented_image = None
                self.dominant_colors = []
                self.masks = []
                self.cluster_visibility = []
                self.show_masks_button.setEnabled(False)
                self.save_button.setEnabled(False)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load image: {str(e)}"
                )

    def display_image(self, image):
        """Displays an image in the GUI with proper scaling and aspect ratio"""
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = channel * width
            
            q_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )
            
            # Calculate scaling while preserving aspect ratio
            label_size = self.image_label.size()
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)

    def show_filters(self):
        """Opens the filter dialog"""
        if self.original_image is None:
            return
            
        dialog = ImageFilterDialog(self.original_image, self)
        if dialog.exec_() == QDialog.Accepted:
            self.original_image = dialog.get_filtered_image()
            self.display_image(self.original_image)
            
            # Reset segmentation results since image has changed
            self.segmented_image = None
            self.dominant_colors = []
            self.masks = []
            self.cluster_visibility = []
            self.show_masks_button.setEnabled(False)
            self.save_button.setEnabled(False)

    def segment_image(self):
        """Performs K-means segmentation with progress feedback"""
        if self.original_image is None:
            return
            
        k = int(self.cluster_input.text())
        
        # Disable UI during processing
        self.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # Prepare data for clustering
            data = self.original_image.reshape((-1, 3))
            data = np.float32(data)
            
            # Configure K-means parameters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            attempts = 10
            
            # Perform K-means clustering
            _, labels, centers = cv2.kmeans(
                data,
                k,
                None,
                criteria,
                attempts,
                cv2.KMEANS_PP_CENTERS
            )
            
            # Convert back to image format
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            self.segmented_image = segmented_data.reshape(self.original_image.shape)
            self.dominant_colors = centers.tolist()
            
            # Generate masks for each cluster
            self.masks = []
            self.cluster_visibility = []
            for i in range(k):
                mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                mask[labels.reshape(self.original_image.shape[:2]) == i] = 255
                self.masks.append(mask)
                self.cluster_visibility.append(True)
            
            # Update display
            self.display_image(self.segmented_image)
            self.show_masks_button.setEnabled(True)
            self.save_button.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Segmentation Error",
                f"Failed to segment image: {str(e)}"
            )
        
        finally:
            # Re-enable UI
            self.setEnabled(True)

    def save_segments(self):
        """Saves the segmented image with error handling"""
        if self.segmented_image is None:
            QMessageBox.warning(
                self,
                "Save Segments",
                "No segmented image to save."
            )
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Segmented Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # Ensure proper file extension
                if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                    file_path += '.png'  # Default to PNG if no extension specified
                
                # Save segmented image
                cv2.imwrite(file_path, self.segmented_image)
                
                # Create directory for individual masks if user wants to save them
                save_masks = QMessageBox.question(
                    self,
                    "Save Individual Masks",
                    "Would you like to save individual cluster masks as well?",
                    QMessageBox.Yes | QMessageBox.No
                ) == QMessageBox.Yes
                
                if save_masks:
                    # Create directory for masks
                    base_path = os.path.splitext(file_path)[0]
                    masks_dir = f"{base_path}_masks"
                    os.makedirs(masks_dir, exist_ok=True)
                    
                    # Save each visible mask
                    for idx, (mask, color, is_visible) in enumerate(zip(
                        self.masks,
                        self.dominant_colors,
                        self.cluster_visibility
                    )):
                        if is_visible:
                            # Create colored mask
                            mask_color = np.zeros_like(self.original_image)
                            mask_color[mask == 255] = color
                            
                            # Save mask
                            mask_path = os.path.join(
                                masks_dir,
                                f"cluster_{idx + 1}.png"
                            )
                            cv2.imwrite(mask_path, mask_color)
                    
                    QMessageBox.information(
                        self,
                        "Save Complete",
                        f"Segmented image and masks saved successfully.\nMasks directory: {masks_dir}"
                    )
                else:
                    QMessageBox.information(
                        self,
                        "Save Complete",
                        "Segmented image saved successfully."
                    )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save image: {str(e)}"
                )

    def show_masks(self):
        """Opens the mask visibility dialog"""
        if not self.masks or not self.dominant_colors:
            QMessageBox.warning(
                self,
                "Show Masks",
                "No segmentation masks available. Please segment the image first."
            )
            return
            
        dialog = UniformColorMaskDialog(
            self.masks,
            self.dominant_colors,
            self.original_image,
            self,
            self.cluster_visibility
        )
        dialog.exec_()



def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    
    # Set application-wide style
    app.setStyle('Fusion')
    
    # Create and set a dark palette for better contrast
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(palette)
    
    window = KMeansSegmentationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()