import numpy as np
from scipy.linalg import block_diag

class QuaternionEKF:
    def __init__(self, process_noise_std, accel_noise_std, gyro_bias_std=0.01):
        # State vector: [qw, qx, qy, qz, bias_x, bias_y, bias_z]
        self.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state (identity quaternion, zero bias)
        
        # State covariance matrix
        self.P = np.eye(7) * 0.01
        
        # Process noise covariance
        quaternion_noise = process_noise_std**2 * np.eye(4)  # Or zeros if not modeling quaternion process noise
        bias_noise = gyro_bias_std**2 * np.eye(3)
        self.Q = block_diag(quaternion_noise, bias_noise) 
        
        # Measurement noise covariance
        self.R_accel = accel_noise_std**2 * np.eye(3)
        
        # Gravity vector in world frame
        self.gravity = np.array([0, 0, 1.0])  # Normalized gravity
        
        # Identity matrix for Kalman gain calculation
        self.I = np.eye(7)

    def normalize_quaternion(self):
        """Normalize the quaternion part of the state vector"""
        q_norm = np.linalg.norm(self.x[0:4])
        if q_norm > 0:
            self.x[0:4] = self.x[0:4] / q_norm

    def quaternion_multiply(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def quaternion_conjugate(self, q):
        """Return the conjugate of quaternion q"""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_rotate_vector(self, q, v):
        """Rotate vector v by quaternion q"""
        q_v = np.array([0, v[0], v[1], v[2]])
        q_conj = self.quaternion_conjugate(q)
        
        # q * q_v * q_conj
        rotated = self.quaternion_multiply(
            self.quaternion_multiply(q, q_v), 
            q_conj
        )
        
        return rotated[1:4]  # Return vector part

    def predict(self, gyro, dt):
        """
        Prediction step of the Kalman filter
        
        Parameters:
        -----------
        gyro : array_like
            Gyroscope measurements in rad/s (3 components)
        dt : float
            Time step in seconds
        """
        # Extract current quaternion and bias
        q = self.x[0:4]
        bias = self.x[4:7]
        
        # Correct gyroscope readings with estimated bias
        gyro_corrected = gyro - bias
        
        # Calculate rotation vector (angular velocity * time)
        theta_vec = gyro_corrected * dt
        
        # Calculate rotation angle
        theta = np.linalg.norm(theta_vec)
        
        # Create rotation quaternion using exponential mapping
        if theta < 1e-10:
            # Small angle approximation
            delta_q = np.array([1, theta_vec[0]/2, theta_vec[1]/2, theta_vec[2]/2])
        else:
            # Normalize axis
            axis = theta_vec / theta
            
            # Calculate half angle
            half_theta = theta * 0.5
            
            # Create quaternion
            delta_q = np.array([
                np.cos(half_theta),
                axis[0] * np.sin(half_theta),
                axis[1] * np.sin(half_theta),
                axis[2] * np.sin(half_theta)
            ])
        
        # Apply rotation: q_new = q ⊗ delta_q
        q_new = self.quaternion_multiply(q, delta_q)
        
        # Bias is assumed constant (random walk model)
        bias_new = bias
        
        # Update state vector
        self.x[0:4] = q_new
        self.x[4:7] = bias_new
        
        # Normalize quaternion
        self.normalize_quaternion()
        
        # Compute Jacobian of the state transition function
        F = self.compute_state_transition_jacobian(q, gyro_corrected, dt)
        
        # Update covariance matrix
        self.P = F @ self.P @ F.T + self.Q
        
    def compute_state_transition_jacobian(self, q, gyro, dt):
        """
        Compute the Jacobian of the state transition function
        
        Parameters:
        -----------
        q : array_like
            Current quaternion (4 components)
        gyro : array_like
            Bias-corrected gyroscope readings (3 components)
        dt : float
            Time step in seconds
        
        Returns:
        --------
        F : ndarray
            State transition Jacobian matrix (7×7)
        """
        qw, qx, qy, qz = q
        wx, wy, wz = gyro
        
        # Calculate rotation angle
        theta = np.linalg.norm(gyro) * dt
        
        if theta < 1e-10:
            # Small angle approximation for quaternion update
            dq_dq = np.eye(4) + 0.5 * dt * np.array([
                [0, -wx, -wy, -wz],
                [wx, 0, wz, -wy],
                [wy, -wz, 0, wx],
                [wz, wy, -wx, 0]
            ])
        else:
            # Use quaternion multiplication matrix of delta_q
            axis = gyro / np.linalg.norm(gyro)
            half_theta = theta * 0.5
            sin_half = np.sin(half_theta)
            cos_half = np.cos(half_theta)
            
            dq_w = cos_half
            dq_x = axis[0] * sin_half
            dq_y = axis[1] * sin_half
            dq_z = axis[2] * sin_half
            
            dq_dq = np.array([
                [dq_w, -dq_x, -dq_y, -dq_z],
                [dq_x, dq_w, -dq_z, dq_y],
                [dq_y, dq_z, dq_w, -dq_x],
                [dq_z, -dq_y, dq_x, dq_w]
            ])
        
        # Compute Jacobian with respect to gyro bias
        G = -0.5 * dt * np.array([
            [0, 0, 0],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])
        
        # Combine to form complete state transition Jacobian
        F = np.zeros((7, 7))
        F[0:4, 0:4] = dq_dq
        F[0:4, 4:7] = G
        F[4:7, 4:7] = np.eye(3)  # Bias transition (identity)
        
        return F
    
    def update_with_accel(self, accel):
        """
        Update state using accelerometer measurements
        
        Parameters:
        -----------
        accel : array_like
            Accelerometer measurements (3 components)
        """
        # Normalize accelerometer reading
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-10:
            return  # Skip update if acceleration is too small
        
        accel_normalized = accel / accel_norm
        
        # Current quaternion
        q = self.x[0:4]
        
        # Predicted gravity direction in body frame
        expected_gravity = self.quaternion_rotate_vector(
            self.quaternion_conjugate(q), 
            self.gravity
        )
        
        # Measurement residual (innovation)
        y = accel_normalized - expected_gravity
        
        # Jacobian of measurement model
        H = self.compute_accel_jacobian(q)
        
        # Kalman update
        self._apply_update(y, H, self.R_accel)
    
    def _apply_update(self, y, H, R):
        """
        Apply Kalman update step
        
        Parameters:
        -----------
        y : ndarray
            Measurement residual (innovation)
        H : ndarray
            Measurement Jacobian
        R : ndarray
            Measurement noise covariance
        """
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update using Joseph form for better numerical stability
        self.P = (self.I - K @ H) @ self.P
        
        # Normalize quaternion part
        self.normalize_quaternion()
    
    def compute_accel_jacobian(self, q):
        """
        Compute Jacobian of accelerometer measurement model
        
        Parameters:
        -----------
        q : array_like
            Current quaternion (4 components)
        
        Returns:
        --------
        H : ndarray
            Measurement Jacobian (3×7)
        """
        qw, qx, qy, qz = q
        
        # Compute derivatives of rotated gravity vector with respect to quaternion
        H_q = 2 * np.array([
            # dg_x/dq
            [qy, qz, qw, qx],
            
            # dg_y/dq
            [-qx, -qw, qz, qy],
            
            # dg_z/dq
            [qw, -qx, -qy, qz]
        ])
        
        # Complete Jacobian with zeros for bias part
        H = np.zeros((3, 7))
        H[:, 0:4] = H_q
        
        return H