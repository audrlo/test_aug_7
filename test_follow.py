from roboclaw import Roboclaw
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class PersonInfo:
    """Information about a detected person"""
    distance: float  # meters from camera
    confidence: float  # YOLOv8 confidence (0.0 to 1.0)
    bbox: List[float]  # [x1, y1, x2, y2] coordinates
    center_x: float  # center of person in image
    center_y: float  # center of person in image

@dataclass
class ObstacleInfo:
    """Information about detected obstacles"""
    distance: float  # meters from camera
    type: str  # object type
    confidence: float  # YOLOv8 confidence
    bbox: List[float]  # [x1, y1, x2, y2] coordinates

class HumanFollowingRobot:
    def __init__(self):
        # Motor controller setup
        self.roboclaw = Roboclaw("/dev/ttyACM0", 38400)
        self.roboclaw.Open()
        self.address = 0x80
        
        # Verify Roboclaw connection
        result, version = self.roboclaw.ReadVersion(self.address)
        print(f"Roboclaw comms: {result}, Version: {version}")
        if not result:
            raise RuntimeError("Failed to connect to Roboclaw")
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        
        # YOLOv8 setup
        self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
        
        # Control parameters (following avoid_obstacle.py conventions)
        self.RAMP_STEP_QPPS = 50
        self.FORWARD_TARGET_QPPS = 50 * 20  # ~"half speed" target
        self.TURN_QPPS = 50 * 10  # turning speed magnitude
        self.RAMP_STEP_DELAY_S = 0.05
        self.OBSTACLE_METERS = 1.0  # minimum safe distance
        self.PERSON_FOLLOW_DISTANCE = 1.0  # target following distance
        self.CHECK_INTERVAL_S = 0.05
        
        # State variables
        self.current_speed_m1 = 0
        self.current_speed_m2 = 0
        self.person_detected = False
        self.last_person_center_x = 320  # center of 640x480 image
        
        print("Human Following Robot initialized successfully")
    
    def get_center_distance(self, depth_frame: rs.depth_frame) -> float:
        """Return distance at image center in meters. Returns float('inf') if invalid."""
        if not depth_frame:
            return float("inf")
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        distance_meters = depth_frame.get_distance(width // 2, height // 2)
        return distance_meters if distance_meters > 0 else float("inf")
    
    def detect_people_and_obstacles(self, rgb_frame, depth_frame) -> Tuple[List[PersonInfo], List[ObstacleInfo]]:
        """Detect people and obstacles using YOLOv8 and depth camera"""
        people = []
        obstacles = []
        
        try:
            # Run YOLOv8 detection
            results = self.yolo_model(rgb_frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        object_type = result.names[int(box.cls)]
                        confidence = float(box.conf)
                        
                        # Calculate object center
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        
                        # Get distance at object center (convert to depth coordinates)
                        depth_x = int(center_x)
                        depth_y = int(center_y)
                        
                        # Ensure coordinates are within depth frame bounds
                        depth_x = max(0, min(depth_x, depth_frame.get_width() - 1))
                        depth_y = max(0, min(depth_y, depth_frame.get_height() - 1))
                        
                        distance = depth_frame.get_distance(depth_x, depth_y)
                        
                        if distance > 0:  # Valid depth reading
                            if object_type == "person":
                                people.append(PersonInfo(
                                    distance=distance,
                                    confidence=confidence,
                                    bbox=bbox.tolist(),
                                    center_x=center_x,
                                    center_y=center_y
                                ))
                            else:
                                obstacles.append(ObstacleInfo(
                                    distance=distance,
                                    type=object_type,
                                    confidence=confidence,
                                    bbox=bbox.tolist()
                                ))
        
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
        
        return people, obstacles
    
    def set_forward_speed(self, qpps: int) -> None:
        """Set forward speed using project's sign convention: M1 forward is +, M2 forward is -"""
        self.roboclaw.SpeedM1(self.address, qpps)
        self.roboclaw.SpeedM2(self.address, -qpps)
        self.current_speed_m1 = qpps
        self.current_speed_m2 = -qpps
    
    def stop_both(self) -> None:
        """Stop both motors"""
        self.roboclaw.SpeedM1(self.address, 0)
        self.roboclaw.SpeedM2(self.address, 0)
        self.current_speed_m1 = 0
        self.current_speed_m2 = 0
    
    def ramp_speed_to_target(self, target_qpps: int, step_qpps: int, step_delay_s: float) -> None:
        """Ramp speed up or down smoothly to target"""
        current_qpps = max(abs(self.current_speed_m1), abs(self.current_speed_m2))
        
        if target_qpps > current_qpps:
            # Ramp up
            for qpps in range(int(current_qpps), target_qpps + 1, step_qpps):
                self.set_forward_speed(qpps)
                time.sleep(step_delay_s)
        else:
            # Ramp down
            for qpps in range(int(current_qpps), target_qpps - 1, -step_qpps):
                self.set_forward_speed(qpps)
                time.sleep(step_delay_s)
    
    def turn_left_until_clear(self, turn_qpps: int, obstacle_meters: float, poll_interval_s: float) -> None:
        """Turn left until no obstacles detected, following avoid_obstacle.py pattern"""
        # Differential turn-in-place left using the project's sign convention
        # Left wheel backward (M1 negative), right wheel forward (M2 negative)
        self.roboclaw.SpeedM1(self.address, -turn_qpps)
        self.roboclaw.SpeedM2(self.address, -turn_qpps)
        print("Turning left until clear")
        time.sleep(1)  # wait 1 second to let the robot turn left
        
        while True:
            frames = self.pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            distance = self.get_center_distance(depth)
            if distance > obstacle_meters:
                break
            time.sleep(poll_interval_s)
        
        self.stop_both()
    
    def follow_person(self, person: PersonInfo) -> None:
        """Follow a detected person maintaining safe distance"""
        distance = person.distance
        center_x = person.center_x
        
        print(f"Following person at {distance:.2f}m, center_x: {center_x:.1f}")
        
        # Distance control
        if distance < self.PERSON_FOLLOW_DISTANCE:
            # Too close - back up
            print("Too close to person, backing up")
            self.roboclaw.SpeedM1(self.address, -self.FORWARD_TARGET_QPPS // 2)
            self.roboclaw.SpeedM2(self.address, self.FORWARD_TARGET_QPPS // 2)
            time.sleep(0.5)  # Back up for 0.5 seconds
            self.stop_both()
            return
        
        elif distance > self.PERSON_FOLLOW_DISTANCE + 0.5:
            # Too far - move forward
            print("Moving toward person")
            self.ramp_speed_to_target(self.FORWARD_TARGET_QPPS, self.RAMP_STEP_QPPS, self.RAMP_STEP_DELAY_S)
        
        # Lateral positioning - keep person centered
        image_center = 320  # center of 640x480 image
        lateral_error = center_x - image_center
        
        if abs(lateral_error) > 50:  # significant lateral offset
            if lateral_error > 0:
                # Person is to the right, turn right
                print("Person to right, turning right")
                self.roboclaw.SpeedM1(self.address, self.TURN_QPPS // 2)
                self.roboclaw.SpeedM2(self.address, -self.TURN_QPPS // 2)
                time.sleep(0.2)
                self.stop_both()
            else:
                # Person is to the left, turn left
                print("Person to left, turning left")
                self.roboclaw.SpeedM1(self.address, -self.TURN_QPPS // 2)
                self.roboclaw.SpeedM2(self.address, -self.TURN_QPPS // 2)
                time.sleep(0.2)
                self.stop_both()
    
    def handle_obstacles(self, obstacles: List[ObstacleInfo]) -> bool:
        """Handle detected obstacles, return True if obstacle avoidance needed"""
        if not obstacles:
            return False
        
        # Find closest obstacle
        closest_obstacle = min(obstacles, key=lambda obj: obj.distance)
        
        if closest_obstacle.distance <= self.OBSTACLE_METERS:
            print(f"Obstacle detected: {closest_obstacle.type} at {closest_obstacle.distance:.2f}m")
            self.stop_both()
            self.turn_left_until_clear(
                self.TURN_QPPS,
                self.OBSTACLE_METERS,
                self.CHECK_INTERVAL_S
            )
            return True
        
        return False
    
    def run(self):
        """Main robot control loop"""
        print("Starting human following robot...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                rgb_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not rgb_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                rgb_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Detect people and obstacles
                people, obstacles = self.detect_people_and_obstacles(rgb_image, depth_frame)
                
                # Handle obstacles first (safety priority)
                obstacle_avoidance_needed = self.handle_obstacles(obstacles)
                
                if obstacle_avoidance_needed:
                    # Obstacle avoidance took priority, continue to next iteration
                    continue
                
                # Handle person following
                if people:
                    # Find closest person
                    closest_person = max(people, key=lambda p: p.confidence)
                    
                    if closest_person.confidence > 0.5:  # confidence threshold
                        self.person_detected = True
                        self.last_person_center_x = closest_person.center_x
                        self.follow_person(closest_person)
                    else:
                        self.person_detected = False
                        self.stop_both()
                else:
                    # No people detected
                    if self.person_detected:
                        print("Person lost, stopping")
                        self.person_detected = False
                        self.stop_both()
                    else:
                        # No person, no obstacles - stay stopped
                        self.stop_both()
                
                # Display debug info
                self.display_debug_info(rgb_image, people, obstacles)
                
                time.sleep(self.CHECK_INTERVAL_S)
        
        except KeyboardInterrupt:
            print("Stopping due to KeyboardInterrupt...")
        finally:
            self.cleanup()
    
    def display_debug_info(self, rgb_image, people: List[PersonInfo], obstacles: List[ObstacleInfo]):
        """Display debug information on the image"""
        # Draw person detections
        for person in people:
            bbox = person.bbox
            cv2.rectangle(rgb_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Person: {person.distance:.2f}m", 
                       (int(bbox[0]), int(bbox[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw obstacle detections
        for obstacle in obstacles:
            bbox = obstacle.bbox
            cv2.rectangle(rgb_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (0, 0, 255), 2)
            cv2.putText(rgb_image, f"{obstacle.type}: {obstacle.distance:.2f}m", 
                       (int(bbox[0]), int(bbox[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw center line
        height, width = rgb_image.shape[:2]
        cv2.line(rgb_image, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        # Draw status text
        status_text = f"People: {len(people)}, Obstacles: {len(obstacles)}"
        cv2.putText(rgb_image, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow('Human Following Robot', rgb_image)
        if cv2.waitKey(1) == ord('q'):
            raise KeyboardInterrupt
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.stop_both()
        except Exception:
            pass
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
        robot = HumanFollowingRobot()
        robot.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
