from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt
import numpy as np
import time
import os
from math import cos, sin, pi

def calculate_direction_extents(volume_view, z_point, num_directions=8):
    """Calculate radial extents in different directions based on volume mask
    
    Args:
        volume_view: The volume view containing the mask data
        z_point: (x,y,z) coordinates of the center point
        num_directions: Number of radial directions to sample (default 8)
    """
    # Get x,y slice at z_point
    z = int(z_point[2])
    x = int(z_point[0])
    y = int(z_point[1])
    
    # Use the lowest resolution level available
    if hasattr(volume_view.volume, 'levels') and len(volume_view.volume.levels) > 0:
        lowest_res_level = volume_view.volume.levels[-1]  # Last level = lowest resolution
        scale = lowest_res_level.scale
        
        # Scale coordinates to match resolution level
        scaled_z = int(z / scale)
        scaled_x = int(x / scale)
        scaled_y = int(y / scale)
        
        # For Zarr volumes, directly access the data array
        if hasattr(lowest_res_level, 'data'):
            slice_data = lowest_res_level.data[scaled_z, :, :]  # Get xy slice
        else:
            return None
            
        if slice_data is None:
            return None
            
        h, w = slice_data.shape
        
        # Calculate extents in num_directions evenly spaced angles
        extents = []
        for i in range(num_directions):
            angle = 2 * pi * i / num_directions
            dx, dy = cos(angle), sin(angle)
            
            # Follow ray until hitting 3x3 area of black pixels or edge
            extent = 0
            cx, cy = scaled_x, scaled_y
            while True:
                cx += dx
                cy += dy
                px, py = int(cx), int(cy)
                
                # Check bounds with 1 pixel padding for 3x3 check
                if px < 1 or px >= w-1 or py < 1 or py >= h-1:
                    break
                    
                # Check if 3x3 area around pixel is all black
                area = slice_data[py-1:py+2, px-1:px+2]
                if np.all(area == 0):
                    break
                    
                extent += 1
                
            # Scale extent back to original resolution
            extents.append(extent * scale)
            
        return extents
        
    return None

def wrap_angle_into_positive_range(angle):
    """
    Utility to wrap an angle into the [0, 2π) range.
    """
    wrapped = angle % (2 * np.pi)
    # Just to be safe if floating ops cause -epsilon,
    # ensure it's in [0, 2π).
    if wrapped < 0:
        wrapped += 2 * np.pi
    return wrapped

def create_swiss_roll_obj(values, umbilicus_points=None, direction_extents=None, timestamp=None):
    """Create a swiss roll OBJ file with direction-based radial constraints,
       starting from a polar coordinate (init_offset, init_angle), ensuring 
       that CCW does not flip the extents inadvertently.
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H:%M:%S", time.gmtime())
        
    # Create output directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    filename = f'temp/swiss_roll_{timestamp}.obj'
    
    # Extract values
    z_max = values['z_max']
    z_min = values['z_min']
    z_step = values['z_step']
    xy_roll_points = values['xy_roll_points']
    x_loc = values['x_loc']
    y_loc = values['y_loc']
    wraps = values['wraps']
    init_offset = values['init_offset']
    init_angle = values['init_angle']
    init_angle = np.radians(init_angle) #convert to radians
    use_umbilicus = values.get('use_umbilicus', False)
    use_mask = values.get('use_mask', False)
    num_directions = values.get('num_directions', 8)
    volume_view = values.get('volume_view', None)
    direction = values.get('direction', 1)  # +1 => CW, -1 => CCW

    # Number of radial directions in your direction_extents
    N = num_directions  
    # The full angle for the entire roll
    final_t = wraps * 2 * np.pi

    # Create z points
    z_points = np.arange(z_min, z_max + z_step, z_step)
    print("z_points", z_points)
    
    # Umbilicus interpolation (if applicable)
    if use_umbilicus and umbilicus_points is not None and len(umbilicus_points) > 1:
        sorted_points = sorted(umbilicus_points, key=lambda p: p[2])
        umbilicus_z = np.array([p[2] for p in sorted_points])
        umbilicus_x = np.array([p[0] for p in sorted_points])
        umbilicus_y = np.array([p[1] for p in sorted_points])
        
        x_positions = np.interp(z_points, umbilicus_z, umbilicus_x)
        y_positions = np.interp(z_points, umbilicus_z, umbilicus_y)
    else:
        x_positions = np.full_like(z_points, x_loc)
        y_positions = np.full_like(z_points, y_loc)

    with open(filename, 'w') as f:
        f.write("# Swiss Roll OBJ File\n")
        
        # For each z-slice
        for z_idx, z in enumerate(z_points):
            # Possibly use a mask-based approach to get direction extents
            if use_mask and volume_view is not None:
                z_direction_extents = calculate_direction_extents(
                    volume_view, (x_positions[z_idx], y_positions[z_idx], z), num_directions
                )
                if z_direction_extents is None:
                    z_direction_extents = direction_extents
            else:
                z_direction_extents = direction_extents
            
            # Linspace from 0..final_t for each "ring" of the swiss roll
            t_values = np.linspace(0, final_t, xy_roll_points)
            
            for t in t_values:
                # Fraction of total wrap
                u = t / final_t
                
                # Combine the param t with init_angle
                raw_angle = t + init_angle

                # 1) Angle for extents lookup => always [0,2π)
                #    If direction is negative, we do the "2π - angle" trick
                if direction >= 0:
                    angle_for_segments = wrap_angle_into_positive_range(raw_angle)
                else:
                    # If direction < 0, invert raw_angle then wrap
                    # (another valid approach: angle_for_segments = wrap_angle_into_positive_range(-raw_angle))
                    # but commonly we do: angle_for_segments = 2π - (raw_angle mod 2π)
                    # for a direct “reverse” indexing.
                    angle_for_segments = wrap_angle_into_positive_range(
                        2*np.pi - (raw_angle % (2*np.pi))
                    )

                # 2) Determine which segment in the direction_extents array
                segment_float = (angle_for_segments / (2*np.pi)) * N
                k = int(np.floor(segment_float))
                w = segment_float - k  # interpolation fraction in [0, 1)

                # radial extents to interpolate
                d0 = z_direction_extents[k % N]
                d1 = z_direction_extents[(k + 1) % N]
                R_max = (1 - w) * d0 + w * d1

                # Actual radius portion from center to R_max
                actual_radius = u * R_max
                
                # Add the initial offset
                total_radius = init_offset + actual_radius

                # 3) Angle for geometry => includes sign of direction
                angle_for_geometry = direction * raw_angle

                # Convert polar -> Cartesian
                x = x_positions[z_idx] + total_radius * np.cos(angle_for_geometry)
                y = y_positions[z_idx] + total_radius * np.sin(angle_for_geometry)

                # Write out the vertex
                f.write(f"v {x} {y} {z}\n")

                # Texture coordinates (u_normalized, v_normalized)
                v_normalized = (z - z_min) / (z_max - z_min)  # z in [0,1]
                u_normalized = u  # radial fraction
                f.write(f"vt {u_normalized} {v_normalized}\n")
        
        # Write faces (two triangles per quad)
        n_z = len(z_points)
        for z_idx in range(n_z - 1):
            for i in range(xy_roll_points - 1):
                v1 = z_idx * xy_roll_points + i + 1
                v2 = v1 + 1
                v3 = v2 + xy_roll_points
                v4 = v1 + xy_roll_points
                f.write(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}\n")
                f.write(f"f {v1}/{v1} {v3}/{v3} {v4}/{v4}\n")

    return filename


class SwissRollDialog(QDialog):
    def __init__(self, parent=None, volume_view=None):
        super().__init__(parent)
        self.setWindowTitle("Create Swiss Roll Fragment")
        self.resize(600, 400)
        self.volume_view = volume_view
        
        layout = QVBoxLayout()
        layout.setSpacing(20) # Add vertical spacing between sections

        # Calculate default width from volume if available
        default_width = 400.0
        if volume_view is not None:
            volume = volume_view.volume
            if volume is not None:
                # Get ~2/3 of volume width
                default_width = (volume.sizes[0] * 2) / 3
        
        # Position controls
        pos_layout = QHBoxLayout()
        
        # Add checkbox for using umbilicus position
        self.use_umbilicus = QCheckBox("Use Umbilicus Position")
        self.use_umbilicus.setEnabled(False)  # Enabled by default
        pos_layout.addWidget(self.use_umbilicus)
        
        # Add checkbox for using mask constraint
        self.use_mask = QCheckBox("Use Mask Constraint")
        self.use_mask.setEnabled(volume_view is not None)
        pos_layout.addWidget(self.use_mask)

        self.folded = QCheckBox("Folded")
        self.folded.setEnabled(False)
        pos_layout.addWidget(self.folded)
        
        pos_layout.addWidget(QLabel("X Location:"))
        self.x_loc = QSpinBox()
        self.x_loc.setRange(0, 100000)
        self.x_loc.setValue(3000)
        self.x_loc.setMinimumWidth(100)
        pos_layout.addWidget(self.x_loc)
        
        pos_layout.addWidget(QLabel("Y Location:"))
        self.y_loc = QSpinBox()
        self.y_loc.setRange(0, 100000)
        self.y_loc.setValue(3000)
        self.y_loc.setMinimumWidth(100)
        pos_layout.addWidget(self.y_loc)
        layout.addLayout(pos_layout)
        
        # Connect checkbox to handler
        self.use_umbilicus.stateChanged.connect(self.onUseUmbilicusChanged)
        
        # Z range
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z Min:"))
        self.z_min = QSpinBox()
        self.z_min.setRange(0, 1000000)
        self.z_min.setValue(0)
        self.z_min.setMinimumWidth(100)
        z_layout.addWidget(self.z_min)
        z_layout.addWidget(QLabel("Z Max:"))
        self.z_max = QSpinBox()
        self.z_max.setRange(0, 1000000)
        self.z_max.setValue(300)
        self.z_max.setMinimumWidth(100)
        z_layout.addWidget(self.z_max)
        layout.addLayout(z_layout)

        # Roll parameters
        params_layout = QHBoxLayout()
        
        # Number of wraps
        params_layout.addWidget(QLabel("Number of Wraps:"))
        self.wraps = QDoubleSpinBox()
        self.wraps.setRange(0.1, 10000.0)
        self.wraps.setValue(100)
        self.wraps.setSingleStep(0.1)
        self.wraps.setMinimumWidth(100)
        params_layout.addWidget(self.wraps)
        
        # Direction (clockwise/counterclockwise)
        params_layout.addWidget(QLabel("Direction:"))
        self.direction = QComboBox()
        self.direction.addItems(["Clockwise", "Counter-clockwise"])
        self.direction.setCurrentIndex(0)
        params_layout.addWidget(self.direction)
        
        # Total width
        params_layout.addWidget(QLabel("Total Width:"))
        self.total_width = QDoubleSpinBox()
        self.total_width.setRange(1.0, 100000.0)
        self.total_width.setValue(default_width)  # Use calculated default
        self.total_width.setSingleStep(1.0)
        self.total_width.setMinimumWidth(100)
        params_layout.addWidget(self.total_width)
        layout.addLayout(params_layout)

        # Points parameters
        points_layout = QHBoxLayout()
        
        # XY plane points
        points_layout.addWidget(QLabel("XY Wrap Points:"))
        self.xy_roll_points = QSpinBox()
        self.xy_roll_points.setRange(10, 10000000)
        self.xy_roll_points.setValue(5000)
        self.xy_roll_points.valueChanged.connect(self.updateTotalPoints)
        self.xy_roll_points.setMinimumWidth(100)
        points_layout.addWidget(self.xy_roll_points)
        
        # Z steps
        points_layout.addWidget(QLabel("Z Step:"))
        self.z_step = QSpinBox()
        self.z_step.setRange(2, 10000)
        self.z_step.setValue(100)
        self.z_step.valueChanged.connect(self.updateTotalPoints)
        self.z_step.setMinimumWidth(100)
        points_layout.addWidget(self.z_step)
        layout.addLayout(points_layout)

        # Direction extents parameters
        extents_layout = QHBoxLayout()
        extents_layout.addWidget(QLabel("Number of Direction Extents:"))
        self.num_directions = QSpinBox()
        self.num_directions.setRange(4, 10000)  
        self.num_directions.setValue(128)  # Default value
        self.num_directions.setMinimumWidth(100)
        self.num_directions.setToolTip("Number of radial directions to sample for mask constraints")
        extents_layout.addWidget(self.num_directions)
        layout.addLayout(extents_layout)


        # Init offset
        init_offset_layout = QHBoxLayout()
        init_offset_layout.addWidget(QLabel("Init Offset:"))
        self.init_offset = QSpinBox()
        self.init_offset.setRange(0, 10000)
        self.init_offset.setValue(180)
        self.init_offset.setMinimumWidth(100)
        self.init_offset.setToolTip("How far the first roll should be from the umbilicus")
        init_offset_layout.addWidget(self.init_offset)
        layout.addLayout(init_offset_layout)

        # Init angle
        init_angle_layout = QHBoxLayout()
        init_angle_layout.addWidget(QLabel("Init Angle:"))
        self.init_angle = QSpinBox()
        self.init_angle.setRange(0, 360)
        self.init_angle.setValue(200)
        self.init_angle.setMinimumWidth(100)
        self.init_angle.setToolTip("The angle of the first roll starting point from the umbilicus, 0 is directly right, direction depends on direction selected")
        init_angle_layout.addWidget(self.init_angle)
        layout.addLayout(init_angle_layout)

        # Fold offset
        fold_layout = QHBoxLayout()
        fold_layout.addWidget(QLabel("Fold Offset:"))
        self.fold_offset = QSpinBox()
        self.fold_offset.setRange(1, 10000)  
        self.fold_offset.setValue(10)  # Default value
        self.fold_offset.setMinimumWidth(100)
        self.fold_offset.setEnabled(False)
        self.fold_offset.setToolTip("How far the second roll should be from the first")
        fold_layout.addWidget(self.fold_offset)
        layout.addLayout(fold_layout)

        # Total points display
        total_points_layout = QHBoxLayout()
        total_points_layout.addWidget(QLabel("Total Points:"))
        self.total_points_label = QLabel("2000")  # Default value
        self.total_points_label.setMinimumWidth(100)
        total_points_layout.addWidget(self.total_points_label)
        layout.addLayout(total_points_layout)
        self.updateTotalPoints()  # Initialize total points display

        # Buttons
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        ok_button.setMinimumWidth(100)
        cancel_button = QPushButton("Cancel") 
        cancel_button.clicked.connect(self.reject)
        cancel_button.setMinimumWidth(100)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addLayout(buttons)
        
        # Add some padding around the edges
        layout.setContentsMargins(20, 20, 20, 20)
        
        self.setLayout(layout)
    
    def updateTotalPoints(self):
        total = self.xy_roll_points.value() * (((self.z_max.value() - self.z_min.value()) / self.z_step.value()) + 1)
        self.total_points_label.setText(str(total))
    
    def getValues(self):
        return {
            'x_loc': self.x_loc.value(),
            'y_loc': self.y_loc.value(),
            'z_min': self.z_min.value(),
            'z_max': self.z_max.value(),
            'wraps': self.wraps.value(),
            'total_width': self.total_width.value(),
            'z_step': self.z_step.value(),
            'xy_roll_points': self.xy_roll_points.value(),
            'use_umbilicus': self.use_umbilicus.isChecked(),
            'use_mask': self.use_mask.isChecked(),
            'folded': self.folded.isChecked(),
            'fold_offset': self.fold_offset.value(),
            'init_angle': self.init_angle.value(),
            'init_offset': self.init_offset.value(),
            'volume_view': self.volume_view,
            'direction': -1 if self.direction.currentText() == "Counter-clockwise" else 1,
            'num_directions': self.num_directions.value()
        }
        
    def setActiveFragment(self, fragment_view):
        """Enable/disable umbilicus checkbox based on active fragment"""
        if fragment_view and hasattr(fragment_view, 'fragment') and hasattr(fragment_view.fragment, 'is_umbilicus'):
            self.use_umbilicus.setEnabled(True)
            self.use_umbilicus.setChecked(True)
            # if self.use_umbilicus.isChecked():
            self.updateFromUmbilicus(fragment_view)
        else:
            self.use_umbilicus.setEnabled(False)
            self.use_umbilicus.setChecked(False)
            
    def onUseUmbilicusChanged(self, state):
        """Handle umbilicus checkbox state changes"""
        if state == Qt.Checked:
            main_window = self.parent()
            active_fragment = main_window.project_view.mainActiveFragmentView()
            if active_fragment:
                self.updateFromUmbilicus(active_fragment)
                
    def updateFromUmbilicus(self, fragment_view):
        """Update X/Y location from umbilicus first point"""
        if fragment_view.manual_points is not None and len(fragment_view.manual_points) > 0:
            # Get first point (lowest Z)
            first_point = fragment_view.manual_points[0]
            self.x_loc.setValue(int(first_point[0]))
            self.y_loc.setValue(int(first_point[1]))

    def getUmbilicusPoints(self):
        """Get umbilicus points if available"""
        if not self.use_umbilicus.isChecked():
            return None
        
        main_window = self.parent()
        active_fragment = main_window.project_view.mainActiveFragmentView()
        if active_fragment and active_fragment.manual_points is not None:
            return active_fragment.manual_points
        return None
