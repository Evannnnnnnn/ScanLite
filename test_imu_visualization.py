import numpy as np
import open3d as o3d
import serial
import time
import math
import copy

# Import the Kalman filter class from our previous code
from kalman_filter_imu import IMUKalmanFilter, process_imu_data

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    # Roll (X-axis rotation)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = R_z * R_y * R_x
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

class IMUVisualizer:
    def __init__(self, port='/dev/ttyUSB0', baud_rate=115200):
        # Initialize serial port
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        
        # Initialize Kalman filter
        self.kf = IMUKalmanFilter(
            accel_noise=0.1,
            gyro_noise=0.01,
            process_noise=0.001
        )
        
        # Initialize Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="IMU Orientation", width=800, height=600)
        
        # Create a template coordinate frame
        self.frame_template = self.create_orientation_frame(size=0.5)
        
        # Initial frame
        self.frame = copy.deepcopy(self.frame_template)
        self.vis.add_geometry(self.frame)
        
        # Data variables
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        
        # Set up rendering options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 5
        
        # Set up view control
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Simulation time variable
        self.sim_time = 0
        
    def create_orientation_frame(self, size=1.0):
        """Create a custom coordinate frame to visualize orientation."""
        # Create a mesh box as the base
        box = o3d.geometry.TriangleMesh.create_box(width=size/5, height=size/5, depth=size/5)
        box.paint_uniform_color([0.8, 0.8, 0.8])
        box.compute_vertex_normals()
        
        # Create axes as cylinders
        cylinder_radius = size/25
        
        # X-axis (red)
        x_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=size)
        x_axis.paint_uniform_color([1, 0, 0])  # Red
        x_axis.compute_vertex_normals()
        x_transform = np.eye(4)
        x_transform[0:3, 0:3] = euler_to_rotation_matrix(0, np.pi/2, 0)
        x_transform[0, 3] = size/2
        x_axis.transform(x_transform)
        
        # Y-axis (green)
        y_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=size)
        y_axis.paint_uniform_color([0, 1, 0])  # Green
        y_axis.compute_vertex_normals()
        y_transform = np.eye(4)
        y_transform[0:3, 0:3] = euler_to_rotation_matrix(np.pi/2, 0, 0)
        y_transform[1, 3] = size/2
        y_axis.transform(y_transform)
        
        # Z-axis (blue)
        z_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=size)
        z_axis.paint_uniform_color([0, 0, 1])  # Blue
        z_axis.compute_vertex_normals()
        z_transform = np.eye(4)
        z_transform[2, 3] = size/2
        z_axis.transform(z_transform)
        
        # Create small spheres for the tips of the axes
        sphere_radius = cylinder_radius * 2
        
        # X-axis tip
        x_tip = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        x_tip.paint_uniform_color([1, 0, 0])  # Red
        x_tip.compute_vertex_normals()
        x_tip_transform = np.eye(4)
        x_tip_transform[0, 3] = size
        x_tip.transform(x_tip_transform)
        
        # Y-axis tip
        y_tip = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        y_tip.paint_uniform_color([0, 1, 0])  # Green
        y_tip.compute_vertex_normals()
        y_tip_transform = np.eye(4)
        y_tip_transform[1, 3] = size
        y_tip.transform(y_tip_transform)
        
        # Z-axis tip
        z_tip = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        z_tip.paint_uniform_color([0, 0, 1])  # Blue
        z_tip.compute_vertex_normals()
        z_tip_transform = np.eye(4)
        z_tip_transform[2, 3] = size
        z_tip.transform(z_tip_transform)
        
        # Combine all geometries
        frame = box
        frame += x_axis
        frame += y_axis
        frame += z_axis
        frame += x_tip
        frame += y_tip
        frame += z_tip
        
        return frame
    
    def update_visualization(self):
        """Update the visualization based on current orientation."""
        # Create rotation matrix from current Euler angles
        rotation_matrix = euler_to_rotation_matrix(self.roll, self.pitch, self.yaw)
        
        # Create a transformation matrix
        transform = np.eye(4)
        transform[0:3, 0:3] = rotation_matrix
        
        # Create a new frame from the template
        new_frame = copy.deepcopy(self.frame_template)
        new_frame.transform(transform)
        
        # Remove the old frame and add the new one
        self.vis.remove_geometry(self.frame)
        self.frame = new_frame
        self.vis.add_geometry(self.frame)
        
        # Update the visualization
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def read_serial_data(self):
        """Read and process one line of data from the serial port."""
        if self.ser is None or not self.ser.is_open:
            return False
            
        if self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                values = list(map(float, line.split(',')))
                
                if len(values) >= 6:
                    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = values[:6]
                    
                    # Process IMU data with Kalman filter
                    roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = process_imu_data(
                        self.kf, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
                    )
                    
                    # Update orientation
                    self.roll = roll
                    self.pitch = pitch
                    self.yaw = yaw
                    
                    # Print current orientation
                    print(f"Roll: {math.degrees(roll):.2f}°, "
                          f"Pitch: {math.degrees(pitch):.2f}°, "
                          f"Yaw: {math.degrees(yaw):.2f}°")
                    
                    return True
            except ValueError as e:
                print(f"Error parsing data: {e}")
            except Exception as e:
                print(f"Serial read error: {e}")
        
        return False
    
    def generate_simulated_data(self):
        """Generate one sample of simulated IMU data."""
        # Update simulation time
        self.sim_time += 0.05
        t = self.sim_time
        
        # Generate simple rotation pattern
        acc_x = math.sin(t * 0.1) * 0.1
        acc_y = math.cos(t * 0.1) * 0.1
        acc_z = -1.0  # Gravity
        
        gyro_x = math.sin(t * 0.5) * 0.2
        gyro_y = math.cos(t * 0.5) * 0.2
        gyro_z = math.sin(t * 0.3) * 0.2
        
        # Process IMU data with Kalman filter
        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = process_imu_data(
            self.kf, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        )
        
        # Update orientation
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
        print(f"[SIM] Roll: {math.degrees(roll):.2f}°, "
              f"Pitch: {math.degrees(pitch):.2f}°, "
              f"Yaw: {math.degrees(yaw):.2f}°")
    
    def run(self):
        """Main loop - single threaded operation."""
        # Try to connect to serial port
        use_simulation = False
        try:
            self.ser = serial.Serial(self.port, self.baud_rate)
            print(f"Connected to {self.port} at {self.baud_rate} baud")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            print("Running with simulated data instead...")
            use_simulation = True
        
        # Main loop
        try:
            print("Press Escape key to exit")
            last_update_time = time.time()
            
            while True:
                # Calculate elapsed time since last update
                current_time = time.time()
                elapsed = current_time - last_update_time
                
                # Control update rate (target ~30 fps)
                if elapsed >= 0.033:  # ~30 fps
                    # Get IMU data - either from serial or simulation
                    if use_simulation:
                        self.generate_simulated_data()
                    else:
                        # If no new data available, use simulation as fallback
                        if not self.read_serial_data():
                            if elapsed >= 0.1:  # Only use sim data if no real data for 100ms
                                self.generate_simulated_data()
                    
                    # Update the visualization
                    self.update_visualization()
                    
                    # Reset timer
                    last_update_time = current_time
                
                # Check if window is still open
                if not self.vis.poll_events():
                    break
                
                # Small sleep to prevent maxing out CPU
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            # Cleanup
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.vis.destroy_window()

if __name__ == "__main__":
    # Create and start the visualizer
    # Change port and baud rate to match your setup
    visualizer = IMUVisualizer(port='/dev/tty.usbmodem2101', baud_rate=9600)
    visualizer.run()