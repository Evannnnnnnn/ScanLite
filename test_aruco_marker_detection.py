import cv2
import numpy as np
import math
import constant
import pyrealsense2 as rs

def detect_aruco_markers_with_distance():
    # Initialize the webcam
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get the color sensor's intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    # Extract camera matrix parameters from intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])
    
    # Extract distortion coefficients
    # RealSense uses Brown-Conrady distortion model (same as OpenCV)
    dist_coeffs = np.array([
        intrinsics.coeffs[0],  # k1
        intrinsics.coeffs[1],  # k2
        intrinsics.coeffs[2],  # p1
        intrinsics.coeffs[3],  # p2
        intrinsics.coeffs[4]   # k3
    ])
    
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    
    # Distortion coefficients (assume no distortion for simplicity)
    dist_coeffs = np.zeros((5, 1))
    
    # Load the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Create parameters for detection
    parameters = cv2.aruco.DetectorParameters()
    
    # Create the detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Marker size in meters (2.2 cm = 0.022 m)
    marker_size = 0.022
    
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert RealSense frame to OpenCV format
        frame = np.asanyarray(color_frame.get_data())
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        # If markers are detected
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate distance for each marker
            mean_tvec = np.zeros((3, 1))
            mean_rvec = np.zeros((3, 1))
            for i, marker_id in enumerate(ids):
                # Get the corners of the marker
                marker_corners = corners[i][0]
                
                # METHOD 1: Using solvePnP to get exact pose
                # This requires camera calibration for accurate results
                
                # Create 3D points of the marker corners in the marker coordinate system
                # For a square marker of size marker_size, the corners are at:
                objPoints = constant.marker_position[marker_id[0]]
                
                # Convert marker corners to the format needed by solvePnP
                marker_corners_float = marker_corners.astype(np.float32)
                
                # Use solvePnP to estimate pose
                _, rvec, tvec = cv2.solvePnP(
                    objPoints, 
                    marker_corners_float, 
                    camera_matrix, 
                    dist_coeffs
                )
                mean_rvec += rvec
                mean_tvec += tvec

            mean_tvec /= i + 1
            mean_rvec /= i + 1
            print(mean_tvec, mean_rvec)

        
        # Display the frame
        cv2.imshow('ArUco Marker Detection with Distance', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_markers_with_distance()