import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
from typing import List, Optional, Tuple
import math


class FallDetection:
    def __init__(self):
        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.pipeline.start(config)
        
        # YOLOv8 setup - using your trained fall detection model
        self.yolo_model = YOLO('fall_detection_model.pt')  # Your trained model
        
        # Fall detection parameters
        self.fall_threshold_ratio = 0.7  # Height/width ratio threshold for fall detection
        self.confidence_threshold = 0.5
        self.last_status_print = 0
        self.status_print_interval = 2.0  # Print status every 2 seconds
        
        print("Fall Detection System initialized")
    
    def get_person_bbox_ratio(self, bbox: List[float]) -> float:
        """Calculate height/width ratio of person bounding box"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0:
            return 0
        
        return height / width
    
    def detect_fall(self, bbox: List[float]) -> bool:
        """Determine if person has fallen based on bounding box ratio"""
        ratio = self.get_person_bbox_ratio(bbox)
        
        # If height/width ratio is low, person is likely lying down/fallen
        # If ratio is high, person is likely standing
        return ratio < self.fall_threshold_ratio
    
    def detect_people_and_falls(self, rgb_frame) -> Tuple[List[dict], bool]:
        """Detect people and falls using your trained model"""
        people = []
        fall_detected = False
        
        try:
            # Run YOLOv8 detection with your trained model
            results = self.yolo_model(rgb_frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        class_id = int(box.cls)
                        class_name = result.names[class_id]
                        confidence = float(box.conf)
                        
                        # Process detections based on your model's classes
                        if confidence > self.confidence_threshold:
                            # Check if this is a fall detection
                            if class_name.lower() == 'fall':
                                fall_detected = True
                            
                            person_info = {
                                'bbox': bbox.tolist(),
                                'confidence': confidence,
                                'class': class_name,
                                'ratio': self.get_person_bbox_ratio(bbox),
                                'has_fallen': class_name.lower() == 'fall'
                            }
                            people.append(person_info)
        
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
        
        return people, fall_detected
    
    def print_status(self, people: List[dict], fall_detected: bool):
        """Print status messages based on detection results"""
        current_time = time.time()
        
        # Only print status occasionally to avoid spam
        if current_time - self.last_status_print < self.status_print_interval:
            return
        
        self.last_status_print = current_time
        
        if not people:
            print("Person is not in frame")
        elif fall_detected:
            print("Person has fallen over.")
        else:
            print("Person is standing")
    
    def display_debug_info(self, rgb_image, people: List[dict], fall_detected: bool):
        """Display debug information on the image"""
        # Draw person detections
        for person in people:
            bbox = person['bbox']
            color = (0, 0, 255) if person['has_fallen'] else (0, 255, 0)  # Red for fallen, green for standing
            label = "FALLEN" if person['has_fallen'] else "STANDING"
            
            cv2.rectangle(rgb_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            cv2.putText(rgb_image, f"{label}: {person['ratio']:.2f}", 
                       (int(bbox[0]), int(bbox[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw status text
        status_text = f"People: {len(people)}, Fall Detected: {fall_detected}"
        cv2.putText(rgb_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow('Fall Detection', rgb_image)
        if cv2.waitKey(1) == ord('q'):
            raise KeyboardInterrupt
    
    def run(self):
        """Main fall detection loop"""
        print("Starting fall detection system...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                rgb_frame = frames.get_color_frame()
                
                if not rgb_frame:
                    continue
                
                # Convert to numpy array
                rgb_image = np.asanyarray(rgb_frame.get_data())
                
                # Detect people and falls
                people, fall_detected = self.detect_people_and_falls(rgb_image)
                
                # Print status
                self.print_status(people, fall_detected)
                
                # Display debug info
                self.display_debug_info(rgb_image, people, fall_detected)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Cleanup completed")


if __name__ == "__main__":
    try:
        detector = FallDetection()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
