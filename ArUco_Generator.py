import cv2
import numpy as np

# Confirm OpenCV version
print(f"Using OpenCV version: {cv2.__version__}")

# Create dictionary - use the 6x6 dictionary as recommended
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Select 4 well-distributed marker IDs to maximize distinctiveness
marker_ids = [23, 75, 127, 200]  

# Marker size in pixels (adjust based on desired print resolution)
marker_size = 400  # High-resolution marker that can be scaled when printing

# Generate and save each marker
for marker_id in marker_ids:
    # Create empty image
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    
    # Generate marker using the OpenCV 4.11.0 API
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, marker_img, 1)
    
    # Add white border (10% of marker size)
    border_size = marker_size // 10
    bordered_img = np.ones((marker_size + 2*border_size, marker_size + 2*border_size), dtype=np.uint8) * 255
    bordered_img[border_size:border_size+marker_size, border_size:border_size+marker_size] = marker_img
    
    # Save the marker
    cv2.imwrite(f"aruco_marker_{marker_id}.png", bordered_img)
    print(f"Generated ArUco marker with ID: {marker_id}")

print("\nAll 4 markers generated successfully!")
print("Each marker has been saved as an individual PNG file with a white border")

# Create a combined visualization (all 4 markers in one image)
combined_img_size = marker_size + 2*border_size
combined_img = np.ones((combined_img_size*2, combined_img_size*2), dtype=np.uint8) * 255

# Place markers in a 2x2 grid
positions = [(0,0), (0,1), (1,0), (1,1)]

for i, marker_id in enumerate(marker_ids):
    # Read the marker image we just saved
    marker = cv2.imread(f"aruco_marker_{marker_id}.png", cv2.IMREAD_GRAYSCALE)
    
    # Calculate position
    row, col = positions[i]
    y_start = row * combined_img_size
    y_end = (row + 1) * combined_img_size
    x_start = col * combined_img_size
    x_end = (col + 1) * combined_img_size
    
    # Place marker in the combined image
    combined_img[y_start:y_end, x_start:x_end] = marker
    
    # Add label
    text_position = (x_start + 20, y_start + 20)
    cv2.putText(combined_img, f"ID: {marker_id}", text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

# Save combined image
cv2.imwrite("aruco_markers_all.png", combined_img)
print("Combined image saved as 'aruco_markers_all.png'")

print("\nPrinting Instructions:")
print("1. For a 2-3cm physical marker, print at 200-300 DPI")
print("2. Ensure 'Scale to Fit' is disabled when printing")
print("3. Measure the printed markers to record their exact physical size")
print("4. Use the exact physical size in your camera calibration for accurate pose estimation")