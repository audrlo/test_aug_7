#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2

# Start pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert depth frame to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Get depth at center pixel
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        center_distance = depth_frame.get_distance(width // 2, height // 2)
        print(f"Depth at center: {center_distance:.2f} meters")

        # Optional: display as color image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.imshow('Depth', depth_colormap)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
