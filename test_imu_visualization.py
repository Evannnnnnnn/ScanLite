import numpy as np
import open3d as o3d
import serial
import time
import math
import copy

# Import the Kalman filter class from our previous code
from kalman_filter_imu import QuaternionEKF

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
    q (array-like): Quaternion in the form [w, x, y, z]
    
    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = q
    
    # Precompute products that are used more than once
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    
    yy = y * y
    yz = y * z
    yw = y * w
    
    zz = z * z
    zw = z * w
    
    # Construct the rotation matrix
    R = np.array([
        [1 - 2 * (yy + zz),    2 * (xy - zw),      2 * (xz + yw)],
        [2 * (xy + zw),        1 - 2 * (xx + zz),  2 * (yz - xw)],
        [2 * (xz - yw),        2 * (yz + xw),      1 - 2 * (xx + yy)]
    ])
    
    return R

class IMUVisualizer:
    def __init__(self, port='/dev/ttyUSB1', baud_rate=9600):
        # Initialize serial port
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        
        # Initialize Kalman filter
        self.kf = QuaternionEKF(
            process_noise_std=0.001,
            accel_noise_std=0.1,
            gyro_bias_std=0.01
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
        self.q = np.array([1, 0, 0, 0])
        
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
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        # You can access the individual axes if needed:
        # x_axis = coordinate_frame.select_by_index([...])  # Would need appropriate indices

        return coordinate_frame
    
    def update_visualization(self):
        """Update the visualization based on current orientation."""
        # Create rotation matrix from current Euler angles
        rotation_matrix = quaternion_to_rotation_matrix(self.q)
        
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
    
    def read_serial_data(self, dt):
        """Read and process one line of data from the serial port."""
        if self.ser is None or not self.ser.is_open:
            return False
            
        if self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                values = list(map(float, line.split(',')))
                
                if len(values) >= 6:
                    acc = values[: 3]
                    gyro = values[3: 6]
                    
                    # Process IMU data with Kalman filter
                    self.kf.predict(gyro, dt)
                    self.kf.update_with_accel(acc)
                    
                    # Update orientation

                    self.q = self.kf.x[: 4]
                    
                    # Print current orientation
                    print(self.q)
                    
                    return True
            except ValueError as e:
                print(f"Error parsing data: {e}")
            except Exception as e:
                print(f"Serial read error: {e}")
        
        return False
    
    def generate_simulated_data(self, dt):
        """Generate one sample of simulated IMU data."""
        print("simulating")
        # Update simulation time
        self.sim_time += 0.05
        t = self.sim_time
        
        # Generate simple rotation pattern
        acc = np.array([math.sin(t * 0.1) * 0.1, math.cos(t * 0.1) * 0.1, 1.0])  # Gravity
        
        gyro = np.array([math.sin(t * 0.5) * 0.2, math.cos(t * 0.5) * 0.2, math.sin(t * 0.3) * 0.2])
        
        # Process IMU data with Kalman filter
        self.kf.predict(gyro, dt)
        self.kf.update_with_accel(acc)
        
        # Update orientation
        self.q = self.kf.x[: 4]
    
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
                        self.generate_simulated_data(elapsed)
                    else:
                        # If no new data available, use simulation as fallback
                        if not self.read_serial_data(elapsed):
                            if elapsed >= 0.1:  # Only use sim data if no real data for 100ms
                                self.generate_simulated_data(elapsed)
                    
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
    visualizer = IMUVisualizer(port='/dev/tty.usbmodem101', baud_rate=9600)
    visualizer.run()