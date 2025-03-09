import open3d as o3d
import numpy as np
import pyrealsense2 as rs
import time

def find_and_fix_empty_point_cloud():
    print("Starting RealSense point cloud debugging...")
    
    # Initialize RealSense pipeline
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable both depth and color streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    except Exception as e:
        print(e)
        return
    
    # Start streaming (with error checking)
    try:
        pipeline_profile = pipeline.start(config)
        print("✅ Pipeline started successfully")
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return
    
    # Get depth scale for distance calculation
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")
    
    # Warmup the camera (very important for RealSense)
    print("Warming up camera (this is crucial)...")
    for i in range(30):
        pipeline.wait_for_frames()
        if i % 10 == 0:
            print(f"  Warmup {i+1}/30")
    
    # Create point cloud object
    pc = rs.pointcloud()
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("RealSense Point Cloud", width=800, height=600)
    
    # Set black background
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 1.0
    
    # Add origin marker for reference
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coord)
    
    # Create empty point cloud
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    # Main processing loop
    try:
        # Keep visualization running
        print("\nPoint cloud visualization running. Press [ESC] in the window to close.")

        frame_count = 0
        while True:  # Try for 100 frames
            # Capture frames
            start = time.time()
            frames = pipeline.wait_for_frames()
            
            # Align depth and color frames
            depth_frame = frames.get_depth_frame()
            
            # Check if frames are valid
            if not depth_frame:
                print("❌ Invalid frames, skipping...")
                continue
            
            # Generate point cloud (with proper setup)
            points = pc.calculate(depth_frame)

            # Get vertices and check
            vertices = np.asarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Set points in Open3D point cloud
            pcd.points = o3d.utility.Vector3dVector(vertices)
            
            # Set all points to white
            pcd.paint_uniform_color([1, 1, 1])
            
            # Update visualization
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # time.sleep(0.05)
            # Short delay between frames
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    find_and_fix_empty_point_cloud()