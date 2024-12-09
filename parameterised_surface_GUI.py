import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets
import zarr
from umbilicus import Umbilicus
import argparse
from vtkmodules.vtkInteractionWidgets import vtkPointHandleRepresentation3D, vtkHandleWidget
from PyQt5 import QtCore
from vtkmodules.vtkRenderingCore import vtkPropPicker

def parse_args():
    parser = argparse.ArgumentParser(description='Swiss Roll Generator')
    parser.add_argument(
        '--zarr-path',
        default="/Users/jamesdarby/Documents/VesuviusScroll/GP/Vesuvius_Data_Download/Scroll1/Scroll1.zarr",
        help='Path to zarr file'
    )
    parser.add_argument(
        '--umbilicus-path',
        default="/Users/jamesdarby/Desktop/test_segs/umbilicii/s1A_54kev_7.91um_zyx_umbilicus_points.txt",
        help='Path to umbilicus points file'
    )
    parser.add_argument(
        '--z-value',
        type=int,
        default=2048*4,
        help='Z-value for slice'
    )
    return parser.parse_args()

def create_swiss_roll(min_t, max_t, min_h, max_h, t_steps, h_steps, frequency=1.0, radius=1.0, x_loc=0, y_loc=0, control_points=None):
    """Generate a Swiss roll mesh based on parameters and control point forces."""
    t = np.linspace(min_t, max_t, t_steps)
    h = np.linspace(min_h, max_h, h_steps)
    T, H = np.meshgrid(t, h)
    
    # Base positions without forces
    x = x_loc + (radius * T * np.cos(frequency * T))
    y = y_loc + (radius * T * np.sin(frequency * T))
    z = H

    if control_points:
        # Convert to numpy arrays for vectorized operations
        points = np.column_stack((x.flatten(), y.flatten()))
        
        # Apply forces from each control point
        for cp in control_points:
            # Calculate distances from control point to all vertices
            cp_pos = np.array([cp.position[0], cp.position[1]])
            
            # Get the direction vector of the major axis (from strength)
            major_axis = np.array([cp.x_strength, cp.y_strength])
            major_length = np.linalg.norm(major_axis)
            if major_length > 0:
                major_dir = major_axis / major_length
            else:
                continue
                
            # Calculate minor axis direction (perpendicular)
            minor_dir = np.array([-major_dir[1], major_dir[0]])
            
            # Transform points to ellipse coordinate system
            diff_vectors = points - cp_pos
            
            # Project onto major and minor axes
            major_proj = np.dot(diff_vectors, major_dir)
            minor_proj = np.dot(diff_vectors, minor_dir)
            
            # Calculate normalized distances in ellipse space
            # Use strength magnitude for major axis and width for minor axis
            major_dist = major_proj / major_length
            minor_dist = minor_proj / (cp.width / 2)  # Divide width by 2 as it's the full width
            
            # Calculate elliptical distance
            elliptical_dist = np.sqrt(major_dist**2 + minor_dist**2)
            
            # Create falloff weight based on elliptical distance
            sigma = 1.0  # This controls how quickly the force falls off
            weights = np.exp(-elliptical_dist**2 / (2 * sigma**2))
            weights = weights.reshape(-1, 1)
            
            # Apply weighted forces
            force = np.array([cp.x_strength, cp.y_strength])
            points += weights * force
        
        # Update x and y with new positions
        x = points[:, 0].reshape(h_steps, t_steps)
        y = points[:, 1].reshape(h_steps, t_steps)

    # Create the grid
    x = x.flatten(order='F')
    y = y.flatten(order='F')
    z = z.flatten(order='F')
    points = np.column_stack((x, y, z))
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [h_steps, t_steps, 1]
    return grid

class ControlPoint:
    def __init__(self, plotter, position=(0, 0, 0), x_strength=100.0, y_strength=100.0):
        self.plotter = plotter
        self.position = np.array(position)
        self.x_strength = x_strength
        self.y_strength = y_strength
        
        # Create sphere for visualization
        self.sphere = pv.Sphere(radius=20, center=position)
        self.sphere_actor = plotter.add_mesh(self.sphere, color='red')
        
        # Add selected state
        self.selected = False
        
        # Create handle widget for point dragging
        self.handle_rep = vtkPointHandleRepresentation3D()
        self.handle_rep.SetWorldPosition(position)
        self.handle_rep.GetProperty().SetColor(1, 0, 0)  # Red color
        self.handle_rep.DragableOn()
        self.handle_rep.SetHandleSize(10)
        
        self.handle_widget = vtkHandleWidget()
        self.handle_widget.SetInteractor(plotter.interactor.GetRenderWindow().GetInteractor())
        self.handle_widget.SetRepresentation(self.handle_rep)
        
        # Add interaction callbacks
        self.handle_widget.AddObserver("InteractionEvent", self.on_interaction)
        self.handle_widget.AddObserver("StartInteractionEvent", self.on_start_interaction)
        
        # Add line visualization
        start_point = np.array([position[0], position[1], 40])  # Z=40
        end_point = np.array([
            position[0] + (x_strength),
            position[1] + (y_strength),
            40
        ])
        self.line = pv.Line(start_point, end_point)
        self.line_actor = plotter.add_mesh(self.line, color='blue', line_width=2)
        
        # Add end sphere and handle for strength control
        self.end_sphere = pv.Sphere(radius=15, center=end_point)
        self.end_sphere_actor = plotter.add_mesh(self.end_sphere, color='blue')
        
        # Create handle widget for strength point dragging
        self.strength_handle_rep = vtkPointHandleRepresentation3D()
        self.strength_handle_rep.SetWorldPosition(end_point)
        self.strength_handle_rep.GetProperty().SetColor(0, 0, 1)  # Blue color
        self.strength_handle_rep.DragableOn()
        self.strength_handle_rep.SetHandleSize(8)
        
        self.strength_handle_widget = vtkHandleWidget()
        self.strength_handle_widget.SetInteractor(plotter.interactor.GetRenderWindow().GetInteractor())
        self.strength_handle_widget.SetRepresentation(self.strength_handle_rep)
        self.strength_handle_widget.AddObserver("InteractionEvent", self.on_strength_interaction)
        self.strength_handle_widget.EnabledOn()
        
        self.width = 100.0  # Add width property
        
        # Add width line and sphere
        self.width_line = pv.Line([0, 0, 0], [0, 0, 0])  # Will update in update_line()
        self.width_line_actor = plotter.add_mesh(self.width_line, color='green', line_width=2)
        
        # Add width control sphere and handle
        self.width_sphere = pv.Sphere(radius=15, center=[0, 0, 0])
        self.width_sphere_actor = plotter.add_mesh(self.width_sphere, color='green')
        
        self.width_handle_rep = vtkPointHandleRepresentation3D()
        self.width_handle_rep.SetWorldPosition([0, 0, 0])
        self.width_handle_rep.GetProperty().SetColor(0, 1, 0)  # Green color
        self.width_handle_rep.DragableOn()
        self.width_handle_rep.SetHandleSize(8)
        
        self.width_handle_widget = vtkHandleWidget()
        self.width_handle_widget.SetInteractor(plotter.interactor.GetRenderWindow().GetInteractor())
        self.width_handle_widget.SetRepresentation(self.width_handle_rep)
        self.width_handle_widget.AddObserver("InteractionEvent", self.on_width_interaction)
        self.width_handle_widget.EnabledOn()

    def on_start_interaction(self, obj, event):
        # Deselect all other points
        for point in self.plotter.parent().control_points:
            if point != self:
                point.set_selected(False)
        self.set_selected(True)
        
        # Update GUI with this point's values
        self.plotter.parent().update_control_point_gui(self)
        
    def on_interaction(self, obj, event):
        new_pos = np.array(self.handle_rep.GetWorldPosition())
        # Force Z coordinate to stay constant
        new_pos[2] = self.position[2]  # Keep original Z position
        
        # Update both the internal position and the handle representation
        self.position = new_pos
        self.handle_rep.SetWorldPosition(new_pos)
        
        # Update sphere position
        self.sphere.points = self.sphere.points + (new_pos - np.array(self.sphere.center))
        
        # Update line position
        self.update_line()
        
        # Also update strength handle when main point moves
        end_point = np.array([
            self.position[0] + self.x_strength,
            self.position[1] + self.y_strength,
            40
        ])
        self.strength_handle_rep.SetWorldPosition(end_point)
        
        # Update the swiss roll mesh
        self.plotter.parent().update_plot()

    def on_strength_interaction(self, obj, event):
        new_pos = np.array(self.strength_handle_rep.GetWorldPosition())
        # Force Z coordinate to stay at 40
        new_pos[2] = 40
        
        # Update strength values based on position relative to control point
        self.x_strength = new_pos[0] - self.position[0]
        self.y_strength = new_pos[1] - self.position[1]
        
        # Update end sphere position
        self.end_sphere.points = self.end_sphere.points + (new_pos - np.array(self.end_sphere.center))
        
        # Update line
        self.update_line()
        
        # Update GUI if this point is selected
        if self.selected:
            self.plotter.parent().update_control_point_gui(self)
        
        # Update the swiss roll mesh
        self.plotter.parent().update_plot()

    def on_width_interaction(self, obj, event):
        new_pos = np.array(self.width_handle_rep.GetWorldPosition())
        new_pos[2] = 40  # Force Z coordinate
        
        # Calculate width from position
        base_pos = np.array([self.position[0], self.position[1], 40])
        width_vector = new_pos - base_pos
        self.width = np.linalg.norm(width_vector)
        
        # Update visuals
        self.update_line()
        
        # Update GUI if selected
        if self.selected:
            self.plotter.parent().update_control_point_gui(self)
        
        # Update the swiss roll mesh
        self.plotter.parent().update_plot()

    def update_line(self):
        start_point = np.array([self.position[0], self.position[1], 40])
        end_point = np.array([
            self.position[0] + (self.x_strength),
            self.position[1] + (self.y_strength),
            40
        ])
        self.line.points[0] = start_point
        self.line.points[1] = end_point
        
        # Also update end sphere and handle position
        self.end_sphere.points = self.end_sphere.points + (end_point - np.array(self.end_sphere.center))
        self.strength_handle_rep.SetWorldPosition(end_point)
        
        # Calculate perpendicular vector for width line
        strength_vector = np.array([self.x_strength, self.y_strength, 0])
        perp_vector = np.array([-strength_vector[1], strength_vector[0], 0])
        perp_vector = perp_vector / np.linalg.norm(perp_vector) * self.width
        
        width_start = np.array([self.position[0], self.position[1], 40])
        width_end = width_start + perp_vector
        
        self.width_line.points[0] = width_start
        self.width_line.points[1] = width_end
        
        # Update width sphere and handle
        self.width_sphere.points = self.width_sphere.points + (width_end - np.array(self.width_sphere.center))
        self.width_handle_rep.SetWorldPosition(width_end)

    def set_selected(self, selected):
        self.selected = selected
        color = (0, 1, 0) if selected else (1, 0, 0)  # Green if selected, red if not
        self.handle_rep.GetProperty().SetColor(*color)
        self.sphere_actor.GetProperty().SetColor(*color)
        
class MainWindow(QtWidgets.QMainWindow):
    """Main window class for the PyQt application."""

    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Swiss Roll Generator")
        self.resize(1200, 800)
        self.first_plot = True
        self.args = args
        
        self.load_static_data()
        self.swiss_roll_actor = None
        self.control_points = []
        self.active_point = None
        self.waiting_for_click = False
        self.init_ui()
        
        # Add click observer
        self.plotter.add_key_event('p', self.toggle_picking_mode)
        self.click_observer = self.plotter.interactor.AddObserver('LeftButtonPressEvent', self.on_click)

    def load_static_data(self):
        """Load the plane and umbilicus data once during initialization"""
        zarr_file = zarr.open(self.args.zarr_path)
        if isinstance(zarr_file, zarr.hierarchy.Group):
            zarr_file = zarr_file[0]
        slice = zarr_file[self.args.z_value, :, :]
        
        umbilicus = Umbilicus(self.args.umbilicus_path)
        self.origin_point = umbilicus.get_point_at_z(self.args.z_value)
        
        # Create the static plane without offset
        self.plane = pv.ImageData(
            dimensions=(slice.shape[0], slice.shape[1], 1),
            origin=(0, 0, 0),
            spacing=(1, 1, 1)
        )
        self.plane.point_data["image"] = slice.flatten(order="F")

    def init_ui(self):
        # Create central widget
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # Create PyVista plotter and add to layout
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)

        # Create controls layout
        controls_layout = QtWidgets.QGridLayout()

        # Create spin boxes for parameters
        self.min_t_box = QtWidgets.QDoubleSpinBox()
        self.min_t_box.setRange(0.0, 200.0)
        self.min_t_box.setValue(0.0)
        self.min_t_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Min t:'), 0, 0)
        controls_layout.addWidget(self.min_t_box, 0, 1)

        self.max_t_box = QtWidgets.QDoubleSpinBox()
        self.max_t_box.setRange(0.1, 20000000000.0)
        self.max_t_box.setValue(20.0)
        self.max_t_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Max t:'), 0, 2)
        controls_layout.addWidget(self.max_t_box, 0, 3)

        self.min_h_box = QtWidgets.QDoubleSpinBox()
        self.min_h_box.setRange(0.0, 20.0)
        self.min_h_box.setValue(0.0)
        self.min_h_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Min h:'), 1, 0)
        controls_layout.addWidget(self.min_h_box, 1, 1)

        self.max_h_box = QtWidgets.QDoubleSpinBox()
        self.max_h_box.setRange(0.1, 20000.0)
        self.max_h_box.setValue(40.0)
        self.max_h_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Max h:'), 1, 2)
        controls_layout.addWidget(self.max_h_box, 1, 3)

        self.t_steps_box = QtWidgets.QSpinBox()
        self.t_steps_box.setRange(10, 1000000)
        self.t_steps_box.setValue(10000)
        self.t_steps_box.setSingleStep(10)
        controls_layout.addWidget(QtWidgets.QLabel('t steps:'), 2, 0)
        controls_layout.addWidget(self.t_steps_box, 2, 1)

        self.h_steps_box = QtWidgets.QSpinBox()
        self.h_steps_box.setRange(1, 1000000)
        self.h_steps_box.setValue(2)
        self.h_steps_box.setSingleStep(10)
        controls_layout.addWidget(QtWidgets.QLabel('h steps:'), 2, 2)
        controls_layout.addWidget(self.h_steps_box, 2, 3)

        self.freq_box = QtWidgets.QDoubleSpinBox()
        self.freq_box.setRange(0.1, 100.0)
        self.freq_box.setValue(1.0)
        self.freq_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Frequency:'), 3, 0)
        controls_layout.addWidget(self.freq_box, 3, 1)

        self.radius_box = QtWidgets.QDoubleSpinBox()
        self.radius_box.setRange(0.01, 100000.0)
        self.radius_box.setValue(1.0)
        self.radius_box.setSingleStep(0.1)
        controls_layout.addWidget(QtWidgets.QLabel('Roll Radius:'), 3, 2)
        controls_layout.addWidget(self.radius_box, 3, 3)

        # Add new primary parameter controls in a new row
        controls_layout.addWidget(QtWidgets.QLabel('--- Primary Parameters ---'), 4, 0, 1, 4)

        self.width_box = QtWidgets.QDoubleSpinBox()
        self.width_box.setRange(1.0, 2000000.0)
        initial_width = self.plane.dimensions[1]*0.75 #approximate initial value
        self.width_box.setValue(initial_width)
        self.width_box.setSingleStep(1.0)
        controls_layout.addWidget(QtWidgets.QLabel('Full Radius Width:'), 5, 0)
        controls_layout.addWidget(self.width_box, 5, 1)

        self.wraps_box = QtWidgets.QDoubleSpinBox()
        self.wraps_box.setRange(0.5, 20000.0)
        self.wraps_box.setValue(100.0)
        self.wraps_box.setSingleStep(0.5)
        controls_layout.addWidget(QtWidgets.QLabel('Num Wraps:'), 5, 2)
        controls_layout.addWidget(self.wraps_box, 5, 3)

        # Add x and y location controls
        self.x_loc_box = QtWidgets.QDoubleSpinBox()
        self.x_loc_box.setRange(-1000000.0, 1000000.0)
        self.x_loc_box.setValue(self.origin_point[2])
        self.x_loc_box.setSingleStep(10.0)
        controls_layout.addWidget(QtWidgets.QLabel('X Location:'), 6, 0)
        controls_layout.addWidget(self.x_loc_box, 6, 1)

        self.y_loc_box = QtWidgets.QDoubleSpinBox()
        self.y_loc_box.setRange(-1000000.0, 1000000.0)
        self.y_loc_box.setValue(self.origin_point[1])
        self.y_loc_box.setSingleStep(10.0)
        controls_layout.addWidget(QtWidgets.QLabel('Y Location:'), 6, 2)
        controls_layout.addWidget(self.y_loc_box, 6, 3)

        # Add recenter camera button
        self.recenter_btn = QtWidgets.QPushButton("Recenter Camera")
        controls_layout.addWidget(self.recenter_btn, 7, 0, 1, 4)
        self.recenter_btn.clicked.connect(self.recenter_camera)

        # Add control point controls
        controls_layout.addWidget(QtWidgets.QLabel('--- Control Points ---'), 8, 0, 1, 4)
        
        self.add_point_btn = QtWidgets.QPushButton("Add Control Point")
        controls_layout.addWidget(self.add_point_btn, 9, 0, 1, 2)
        self.add_point_btn.clicked.connect(self.add_control_point)
        
        self.x_strength = QtWidgets.QDoubleSpinBox()
        self.x_strength.setRange(-100000.0, 100000.0)
        self.x_strength.setValue(100.0)
        self.x_strength.valueChanged.connect(self.update_selected_point)
        controls_layout.addWidget(QtWidgets.QLabel('X Strength:'), 10, 0)
        controls_layout.addWidget(self.x_strength, 10, 1)
        
        self.y_strength = QtWidgets.QDoubleSpinBox()
        self.y_strength.setRange(-100000.0, 100000.0)
        self.y_strength.setValue(100.0)
        self.y_strength.valueChanged.connect(self.update_selected_point)
        controls_layout.addWidget(QtWidgets.QLabel('Y Strength:'), 10, 2)
        controls_layout.addWidget(self.y_strength, 10, 3)
        
        # Add width control after x and y strength controls
        self.width_control = QtWidgets.QDoubleSpinBox()
        self.width_control.setRange(0.0, 100000.0)
        self.width_control.setValue(100.0)
        self.width_control.valueChanged.connect(self.update_selected_point)
        controls_layout.addWidget(QtWidgets.QLabel('Width:'), 11, 0)
        controls_layout.addWidget(self.width_control, 11, 1)
        
        vlayout.addLayout(controls_layout)
        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # Connect signals to slots
        self.min_t_box.valueChanged.connect(self.update_min_t)
        self.max_t_box.valueChanged.connect(self.update_max_t)
        self.min_h_box.valueChanged.connect(self.update_min_h)
        self.max_h_box.valueChanged.connect(self.update_max_h)
        self.t_steps_box.valueChanged.connect(self.update_plot)
        self.h_steps_box.valueChanged.connect(self.update_plot)
        self.freq_box.valueChanged.connect(self.update_plot)
        self.radius_box.valueChanged.connect(self.update_plot)
        self.width_box.valueChanged.connect(self.update_derived_params)
        self.wraps_box.valueChanged.connect(self.update_derived_params)
        self.x_loc_box.valueChanged.connect(self.update_plot)
        self.y_loc_box.valueChanged.connect(self.update_plot)
        self.recenter_btn.clicked.connect(self.recenter_camera)
        self.add_point_btn.clicked.connect(self.add_control_point)

        # After connecting all signals, trigger initial parameter update
        self.update_derived_params()

    def update_min_t(self, value):
        self.max_t_box.setMinimum(value + 0.1)
        self.update_plot()

    def update_max_t(self, value):
        self.min_t_box.setMaximum(value - 0.1)
        self.update_plot()

    def update_min_h(self, value):
        self.max_h_box.setMinimum(value + 0.1)
        self.update_plot()

    def update_max_h(self, value):
        self.min_h_box.setMaximum(value - 0.1)
        self.update_plot()

    def calculate_params(self, width, num_wraps):
        """Calculate max_t, frequency, and radius from width and num_wraps"""
        max_t = 2 * np.pi * num_wraps
        radius = width / (2 * max_t)
        frequency = 1.0
        return max_t, frequency, radius

    def update_derived_params(self):
        """Update technical parameters when primary parameters change"""
        width = self.width_box.value()
        num_wraps = self.wraps_box.value()
        max_t, frequency, radius = self.calculate_params(width, num_wraps)
        
        # Block signals to prevent recursive updates
        self.max_t_box.blockSignals(True)
        self.freq_box.blockSignals(True)
        self.radius_box.blockSignals(True)
        
        # Update the technical parameter boxes
        self.max_t_box.setValue(max_t)
        self.freq_box.setValue(frequency)
        self.radius_box.setValue(radius)
        
        # Unblock signals
        self.max_t_box.blockSignals(False)
        self.freq_box.blockSignals(False)
        self.radius_box.blockSignals(False)
        
        self.update_plot()

    def update_plot(self):
        # Store current camera position before updates
        camera_position = self.plotter.camera_position if not self.first_plot else None
        
        min_t = self.min_t_box.value()
        max_t = self.max_t_box.value()
        min_h = self.min_h_box.value()
        max_h = self.max_h_box.value()
        t_steps = self.t_steps_box.value()
        h_steps = self.h_steps_box.value()
        frequency = self.freq_box.value()
        radius = self.radius_box.value()
        x_loc = self.x_loc_box.value()
        y_loc = self.y_loc_box.value()

        grid = create_swiss_roll(min_t, max_t, min_h, max_h, t_steps, h_steps, 
                               frequency, radius, x_loc, y_loc, self.control_points)
        
        if self.swiss_roll_actor is None:
            # First time setup
            self.plotter.clear()
            self.swiss_roll_actor = self.plotter.add_mesh(grid, show_edges=True, color="lightblue", reset_camera=False)
            self.plotter.add_mesh(self.plane, cmap='gray', scalars="image", show_scalar_bar=False, reset_camera=False)
            bounds = grid.bounds
            self.plotter.add_ruler(
                pointa=[bounds[0], bounds[2], bounds[4]],
                pointb=[bounds[1], bounds[2], bounds[4]],
                title="Width"
            )
            self.plotter.add_axes()
            # Set initial camera position
            self.plotter.camera_position = [
                (x_loc, y_loc, 1000),  # Camera position
                (x_loc, y_loc, 0),     # Focal point
                (0, 1, 0)              # Up vector
            ]
            self.plotter.reset_camera()
            self.first_plot = False
        else:
            # Just update the Swiss roll mesh
            self.swiss_roll_actor.GetMapper().SetInputData(grid)
            
        # Restore camera position if it was stored
        if camera_position is not None:
            self.plotter.camera_position = camera_position
        
        self.plotter.render()

    def recenter_camera(self):
        """Recenter the camera on current x,y location"""
        x_loc = self.x_loc_box.value()
        y_loc = self.y_loc_box.value()
        self.plotter.camera_position = [
            (x_loc, y_loc, 1000),  # Camera position
            (x_loc, y_loc, 0),     # Focal point
            (0, 1, 0)              # Up vector
        ]
        self.plotter.reset_camera()

    def add_control_point(self):
        self.waiting_for_click = True
        self.plotter.interactor.GetRenderWindow().SetCurrentCursor(QtCore.Qt.CrossCursor)

    def on_click(self, *args):
        if not self.waiting_for_click:
            return

        # Create a prop picker
        picker = vtkPropPicker()
        
        # Get current mouse position and pick at that location
        click_pos = self.plotter.iren.get_event_position()
        picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)
        world_pos = picker.GetPickPosition()

        # Disable picking mode before creating point
        self.waiting_for_click = False
        self.plotter.interactor.GetRenderWindow().SetCurrentCursor(QtCore.Qt.ArrowCursor)
        
        point = ControlPoint(
            self.plotter,
            position=(world_pos[0], world_pos[1], 0),  # Set Z to 0
            x_strength=self.x_strength.value(),
            y_strength=self.y_strength.value()
        )
        # Add reference to parent window
        self.plotter.parent = lambda: self
        self.control_points.append(point)
        point.handle_widget.EnabledOn()
        self.update_plot()

    def toggle_picking_mode(self):
        if self.waiting_for_click:
            self.waiting_for_click = False
            self.plotter.interactor.GetRenderWindow().SetCurrentCursor(QtCore.Qt.ArrowCursor)
        else:
            self.waiting_for_click = True
            self.plotter.interactor.GetRenderWindow().SetCurrentCursor(QtCore.Qt.CrossCursor)

    def update_control_point_gui(self, point):
        """Update GUI controls with selected point's values"""
        self.x_strength.blockSignals(True)
        self.y_strength.blockSignals(True)
        
        self.x_strength.setValue(point.x_strength)
        self.y_strength.setValue(point.y_strength)
        
        self.x_strength.blockSignals(False)
        self.y_strength.blockSignals(False)
        self.width_control.blockSignals(True)
        self.width_control.setValue(point.width)
        self.width_control.blockSignals(False)

    def update_selected_point(self):
        """Update selected point's values from GUI"""
        selected_points = [p for p in self.control_points if p.selected]
        if selected_points:
            point = selected_points[0]
            point.x_strength = self.x_strength.value()
            point.y_strength = self.y_strength.value()
            point.width = self.width_control.value()
            point.update_line()
            self.update_plot()

if __name__ == '__main__':
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(args)
    window.show()
    sys.exit(app.exec_())
