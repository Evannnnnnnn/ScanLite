import cv2
import numpy as np
import math
import constant

def detect_aruco_markers_with_distance():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get camera matrix (intrinsic parameters)
    # For accurate distance measurement, you should calibrate your camera
    # and use the actual values. These are approximate values for a 640x480 webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Approximate focal length (can be calibrated for better accuracy)
    # Typical webcam focal length is around 500-600 pixels for 640x480 resolution
    focal_length = 550  # This is an approximate value
    
    # Camera matrix
    camera_matrix = np.array([
        [focal_length, 0, frame_width/2],
        [0, focal_length, frame_height/2],
        [0, 0, 1]
    ])
    
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
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # If markers are detected
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate distance for each marker
            for i, marker_id in enumerate(ids):
                print(marker_id)
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
                retval, rvec, tvec = cv2.solvePnP(
                    objPoints, 
                    marker_corners_float, 
                    camera_matrix, 
                    dist_coeffs
                )

                print(tvec)
                
                # Display information on the frame
                # cv2.putText(frame, f"ID: {marker_id[0]}", 
                #             (center[0], center[1] - 40), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"Dist(PnP): {distance_pnp:.2f}m", 
                #             (center[0], center[1] - 20), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"Dist(Simple): {distance_simple:.2f}m", 
                #             (center[0], center[1]), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.circle(frame, center, 4, (0, 0, 255), -1)
                
                # # Print to console
                # print(f"Marker ID: {marker_id[0]}, Distance(PnP): {distance_pnp:.3f}m, Distance(Simple): {distance_simple:.3f}m")
        
        # Display the frame
        cv2.imshow('ArUco Marker Detection with Distance', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def calibrate_camera():
    """
    Camera calibration function - for more accurate measurements,
    you should calibrate your camera using a chessboard pattern
    and replace the approximate camera_matrix values with calibrated ones.
    
    Example calibration code not shown here for brevity.
    """
    pass

if __name__ == "__main__":
    detect_aruco_markers_with_distance()