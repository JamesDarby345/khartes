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

def create_swiss_roll_obj(values, umbilicus_points=None, direction_extents=None, timestamp=None):
    """Create a swiss roll OBJ file with direction-based radial constraints."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H:%M:%S", time.gmtime())
        
    # Create output directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    filename = f'temp/swiss_roll_{timestamp}.obj'
    
    # Extract values
    z_max = values['z_max']
    z_min = values['z_min']
    z_step = values['z_step']
    xy_points = values['xy_points']
    x_loc = values['x_loc']
    y_loc = values['y_loc']
    wraps = values['wraps']
    use_umbilicus = values.get('use_umbilicus', False)
    use_mask = values.get('use_mask', False)
    num_directions = values.get('num_directions', 8)
    volume_view = values.get('volume_view', None)

    # Number of directional constraints
    N = num_directions  # Number of directions to sample
    # The full angle corresponding to the final wrap
    final_t = wraps * 2 * np.pi

    # Create z points
    z_points = np.arange(z_min, z_max + z_step, z_step)
    print("z_points", z_points)
    
    # If using umbilicus, interpolate x,y positions for each z
    if use_umbilicus and umbilicus_points is not None and len(umbilicus_points) > 1:
        # Sort umbilicus points by z coordinate
        sorted_points = sorted(umbilicus_points, key=lambda p: p[2])
        umbilicus_z = np.array([p[2] for p in sorted_points])
        umbilicus_x = np.array([p[0] for p in sorted_points])
        umbilicus_y = np.array([p[1] for p in sorted_points])
        
        # Interpolate x,y positions for each z point
        x_positions = np.interp(z_points, umbilicus_z, umbilicus_x)
        y_positions = np.interp(z_points, umbilicus_z, umbilicus_y)
    else:
        # Use single x,y position for all z points
        x_positions = np.full_like(z_points, x_loc)
        y_positions = np.full_like(z_points, y_loc)

    # Get direction multiplier (-1 for CCW, 1 for CW)
    direction = values.get('direction', 1)
    
    with open(filename, 'w') as f:
        f.write("# Swiss Roll OBJ File\n")
        
        # For each z level
        for z_idx, z in enumerate(z_points):
            # Get direction extents for this z level if using mask
            if use_mask and volume_view is not None:
                z_direction_extents = calculate_direction_extents(volume_view, 
                    (x_positions[z_idx], y_positions[z_idx], z), num_directions)
                if z_direction_extents is None:
                    z_direction_extents = direction_extents
            else:
                z_direction_extents = direction_extents
                
            # Parameter t goes from 0 to final_t for xy_points samples
            t_values = np.linspace(0, wraps * 2 * np.pi, xy_points)
            
            for t in t_values:
                # Compute normalized layer fraction (0 at center, 1 at outer boundary)
                u = t / final_t

                # Current angle on [0,2π)
                angle = t % (2*np.pi)
                if direction < 0:  # CCW case
                    angle = (2*np.pi - angle) % (2*np.pi)  # Reverse angle for CCW

                # Determine which segment of direction_extents we're in
                # segment index scaled by angle/2π * N
                segment_float = (angle / (2*np.pi)) * N
                k = int(np.floor(segment_float))
                w = segment_float - k  # interpolation weight

                # Wrap indices (in case angle is near 2π)
                d0 = z_direction_extents[k % N]
                d1 = z_direction_extents[(k+1) % N]

                # Interpolated max radius for this angle
                R_max = (1 - w) * d0 + w * d1

                # Actual radius at this point
                actual_radius = u * R_max

                # Compute coordinates - multiply by direction for CW/CCW
                x = x_positions[z_idx] + actual_radius * np.cos(direction * t)
                y = y_positions[z_idx] + actual_radius * np.sin(direction * t)

                # Write vertex
                f.write(f"v {x} {y} {z}\n")

                # Write texture coordinates - similar normalization
                v_normalized = (z - z_min) / (z_max - z_min)  # z -> [0,1]
                u_normalized = u  # radius fraction as texture u-coordinate
                f.write(f"vt {u_normalized} {v_normalized}\n")
        
        # Write faces as before
        for z_idx in range(len(z_points)-1):
            for i in range(xy_points-1):
                v1 = z_idx * xy_points + i + 1
                v2 = v1 + 1
                v3 = v2 + xy_points
                v4 = v1 + xy_points
                # Two triangles per quad
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
        self.z_max.setValue(1000)
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
        points_layout.addWidget(QLabel("XY Plane Points:"))
        self.xy_points = QSpinBox()
        self.xy_points.setRange(10, 10000000)
        self.xy_points.setValue(10000)
        self.xy_points.valueChanged.connect(self.updateTotalPoints)
        self.xy_points.setMinimumWidth(100)
        points_layout.addWidget(self.xy_points)
        
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
        self.num_directions.setValue(64)  # Default value
        self.num_directions.setMinimumWidth(100)
        self.num_directions.setToolTip("Number of radial directions to sample for mask constraints")
        extents_layout.addWidget(self.num_directions)
        layout.addLayout(extents_layout)

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
        total = self.xy_points.value() * (self.z_max.value() - self.z_min.value()) / self.z_step.value()
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
            'xy_points': self.xy_points.value(),
            'use_umbilicus': self.use_umbilicus.isChecked(),
            'use_mask': self.use_mask.isChecked(),
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
