import numpy as np
import serial
import time
import math

class IMUKalmanFilter:
    """
    A Kalman filter implementation for IMU data that properly fuses
    accelerometer and gyroscope measurements, including yaw estimation.
    
    State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
    
    Note: Since yaw cannot be derived from accelerometer, it will be 
    estimated only from gyroscope integration and will drift over time.
    """
    def __init__(self, accel_noise=0.1, gyro_noise=0.01, process_noise=0.001):
        # State vector [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.state = np.zeros((6, 1))
        
        # Error covariance matrix - initial uncertainty
        self.P = np.eye(6)
        # Higher initial uncertainty for yaw since we can't measure it directly
        self.P[2, 2] = 10.0
        
        # State transition matrix (for constant angular velocity model)
        self.F = np.eye(6)
        # Will be updated with dt in predict step
        
        # Process noise covariance matrix
        self.Q = np.eye(6) * process_noise
        # Higher process noise for yaw to reflect greater uncertainty
        self.Q[2, 2] = process_noise * 5
        
        # Measurement matrix - maps state to measurements
        # Note: We have 5 measurements (2 from accel, 3 from gyro) but 6 state variables
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # accel_roll = roll
            [0, 1, 0, 0, 0, 0],  # accel_pitch = pitch
            [0, 0, 0, 1, 0, 0],  # gyro_roll_rate = roll_rate
            [0, 0, 0, 0, 1, 0],  # gyro_pitch_rate = pitch_rate
            [0, 0, 0, 0, 0, 1]   # gyro_yaw_rate = yaw_rate
        ])
        
        # Measurement noise covariance matrix
        self.R = np.diag([accel_noise, accel_noise, gyro_noise, gyro_noise, gyro_noise])
        
        # Time tracking
        self.dt = 0.01  # Default time step, will be updated
        self.last_time = None
    
    def update_dt(self):
        """Update the time step based on real elapsed time."""
        current_time = time.time()
        if self.last_time is not None:
            self.dt = current_time - self.last_time
        self.last_time = current_time
        
        # Update state transition matrix with new dt
        self.F[0, 3] = self.dt  # roll += roll_rate * dt
        self.F[1, 4] = self.dt  # pitch += pitch_rate * dt
        self.F[2, 5] = self.dt  # yaw += yaw_rate * dt
    
    def predict(self):
        """Prediction step of the Kalman filter."""
        # Update time step
        self.update_dt()
        
        # Predict state
        self.state = np.dot(self.F, self.state)
        
        # Wrap yaw angle to [-π, π]
        self.state[2, 0] = wrap_angle(self.state[2, 0])
        
        # Predict error covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.state
    
    def update(self, measurement):
        """Update step of the Kalman filter using the measurement vector."""
        # Calculate Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state with measurement
        y = measurement - np.dot(self.H, self.state)  # Measurement residual
        self.state = self.state + np.dot(K, y)
        
        # Wrap yaw angle to [-π, π]
        self.state[2, 0] = wrap_angle(self.state[2, 0])
        
        # Update error covariance
        I = np.eye(self.state.shape[0])
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
                        (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        
        return self.state

def wrap_angle(angle):
    """Wrap angle to range [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

def calculate_angles_from_accelerometer(acc_x, acc_y, acc_z):
    """
    Calculate roll and pitch angles from accelerometer data.
    Roll is rotation around X-axis, pitch is rotation around Y-axis.
    
    Returns:
        tuple: (roll, pitch) in radians
    """
    # Calculate roll (rotation around X-axis)
    roll = math.atan2(acc_y, acc_z)
    
    # Calculate pitch (rotation around Y-axis)
    pitch = math.atan2(-acc_x, math.sqrt(acc_y**2 + acc_z**2))
    
    return roll, pitch

def process_imu_data(filter, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """
    Process IMU data using the Kalman filter.
    
    Args:
        filter: The IMUKalmanFilter instance
        acc_x, acc_y, acc_z: Accelerometer readings in g's
        gyro_x, gyro_y, gyro_z: Gyroscope readings in radians/second
        
    Returns:
        tuple: (roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate)
    """
    # First, predict the next state
    filter.predict()
    
    # Calculate angles from accelerometer
    accel_roll, accel_pitch = calculate_angles_from_accelerometer(acc_x, acc_y, acc_z)
    
    # Extract angular rates directly from gyroscope
    # Note: Depending on your sensor's orientation, you might need to remap axes
    gyro_roll_rate = gyro_x   # Roll rate around x-axis
    gyro_pitch_rate = gyro_y  # Pitch rate around y-axis
    gyro_yaw_rate = gyro_z    # Yaw rate around z-axis
    
    # Create measurement vector
    # Note: We don't have a direct measurement for yaw
    measurement = np.array([
        [accel_roll], 
        [accel_pitch], 
        [gyro_roll_rate], 
        [gyro_pitch_rate],
        [gyro_yaw_rate]
    ])
    
    # Update the filter with all measurements at once
    updated_state = filter.update(measurement)
    
    # Extract values from updated state
    roll = updated_state[0, 0]
    pitch = updated_state[1, 0]
    yaw = updated_state[2, 0]
    roll_rate = updated_state[3, 0]
    pitch_rate = updated_state[4, 0]
    yaw_rate = updated_state[5, 0]
    
    return roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate

def read_imu_from_serial(port='/dev/ttyUSB0', baud_rate=115200):
    """
    Read IMU data from serial port and apply Kalman filter.
    
    This is a template - modify according to your actual serial data format.
    """
    # Initialize serial connection
    ser = serial.Serial(port, baud_rate)
    
    # Initialize Kalman filter with appropriate noise parameters
    # These values should be tuned based on your sensors' characteristics
    kf = IMUKalmanFilter(
        accel_noise=0.1,    # Accelerometer is relatively noisy
        gyro_noise=0.01,    # Gyroscope is more precise but drifts
        process_noise=0.001  # How quickly the state can change
    )
    
    try:
        while True:
            # Read a line from serial port
            line = ser.readline().decode('utf-8').strip()
            
            # Parse the line - adjust this based on your data format
            # Example format: "acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z"
            try:
                values = list(map(float, line.split(',')))
                if len(values) >= 6:
                    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = values[:6]
                    
                    # Apply Kalman filter
                    roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = process_imu_data(
                        kf, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
                    )
                    
                    # Convert from radians to degrees for display
                    roll_deg = math.degrees(roll)
                    pitch_deg = math.degrees(pitch)
                    yaw_deg = math.degrees(yaw)
                    
                    print(f"Roll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")
                    print(f"Rates: {math.degrees(roll_rate):.2f}°/s, "
                          f"{math.degrees(pitch_rate):.2f}°/s, "
                          f"{math.degrees(yaw_rate):.2f}°/s")
            
            except ValueError as e:
                print(f"Error parsing line: {e}")
                continue
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        ser.close()

if __name__ == "__main__":
    # Change port and baud rate to match your setup
    read_imu_from_serial(port='/dev/tty.usbmodem101', baud_rate=9600)