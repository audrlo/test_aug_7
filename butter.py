from roboclaw import Roboclaw
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
import threading
import glob
import os

class RoboClawConnectionManager:
    """Manages RoboClaw connection with automatic reconnection"""
    
    def __init__(self, port="/dev/ttyACM0", baudrate=38400, address=0x80, heartbeat_interval=1.0):
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.heartbeat_interval = heartbeat_interval
        self.roboclaw = None
        self.connected = False
        self.heartbeat_thread = None
        self.stop_heartbeat = False
        
    def find_roboclaw_ports(self):
        """Find available RoboClaw ports"""
        ports = []
        for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
            ports.extend(glob.glob(pattern))
        return sorted(ports)
    
    def test_roboclaw_connection(self, port):
        """Test if a port has a RoboClaw"""
        try:
            if not self.check_port_permissions(port):
                return False
                
            test_roboclaw = Roboclaw(port, self.baudrate)
            test_roboclaw.Open()
            
            # Test communication
            result, version = test_roboclaw.ReadVersion(self.address)
            if result:
                print(f"Found RoboClaw on {port}: {version}")
                test_roboclaw._port.close()
                return True
            else:
                test_roboclaw._port.close()
                return False
        except Exception as e:
            print(f"Error testing {port}: {e}")
            return False
    
    def check_port_permissions(self, port):
        """Check if we have read/write permissions on the port"""
        try:
            return os.access(port, os.R_OK | os.W_OK)
        except:
            return False
    
    def scan_and_connect(self):
        """Scan all ports and connect to first working RoboClaw"""
        ports = self.find_roboclaw_ports()
        print(f"Scanning ports: {ports}")
        
        for port in ports:
            if self.test_roboclaw_connection(port):
                self.port = port
                print(f"Selected port: {self.port}")
                return True
        return False
    
    def wait_for_connection(self, max_retries=None, retry_delay=2.0):
        """Wait for RoboClaw connection"""
        retry_count = 0
        
        while max_retries is None or retry_count < max_retries:
            try:
                # Try original port first
                if self.test_roboclaw_connection(self.port):
                    self.roboclaw = Roboclaw(self.port, self.baudrate)
                    self.roboclaw.Open()
                    self.connected = True
                    print(f"Connected to RoboClaw on {self.port}")
                    return True
                
                # If original port failed, scan for new port
                print(f"Port {self.port} failed, scanning for new port...")
                if self.scan_and_connect():
                    self.roboclaw = Roboclaw(self.port, self.baudrate)
                    self.roboclaw.Open()
                    self.connected = True
                    print(f"Connected to RoboClaw on {self.port}")
                    return True
                
                print(f"Connection attempt {retry_count + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_count += 1
                
            except Exception as e:
                print(f"Connection error: {e}")
                retry_count += 1
                time.sleep(retry_delay)
        
        print("Failed to connect to RoboClaw")
        return False
    
    def start_heartbeat(self):
        """Start heartbeat monitoring thread"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.stop_heartbeat = False
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
            print("Heartbeat monitoring started")
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while not self.stop_heartbeat:
            try:
                if self.roboclaw and self.connected:
                    result, version = self.roboclaw.ReadVersion(self.address)
                    if not result:
                        print("RoboClaw heartbeat failed - connection lost")
                        self.connected = False
                        self.roboclaw._port.close()
                        self.roboclaw = None
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                print(f"Heartbeat error: {e}")
                self.connected = False
                if self.roboclaw:
                    try:
                        self.roboclaw._port.close()
                    except:
                        pass
                    self.roboclaw = None
                time.sleep(self.heartbeat_interval)
    
    def ensure_connection(self):
        """Ensure connection is active, reconnect if needed"""
        if not self.connected or not self.roboclaw:
            print("RoboClaw disconnected, attempting to reconnect...")
            return self.wait_for_connection()
        return True
    
    def is_connected(self):
        """Check if currently connected"""
        return self.connected and self.roboclaw is not None
    
    def get_roboclaw(self):
        """Get RoboClaw instance if connected"""
        return self.roboclaw if self.connected else None
    
    def close(self):
        """Close connection and stop heartbeat"""
        self.stop_heartbeat = True
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1.0)
        
        if self.roboclaw:
            try:
                self.roboclaw._port.close()
            except:
                pass
            self.roboclaw = None
        
        self.connected = False
        print("RoboClaw connection closed")

class HumanFollowingRobot:
    """Simple human following robot with obstacle avoidance"""
    
    def __init__(self):
        # RoboClaw connection
        self.connection_manager = RoboClawConnectionManager()
        
        # Motor control parameters (cut in half for smoother movement)
        self.FORWARD_TARGET_QPPS = 500   # Base forward speed (was 1000)
        self.TURN_QPPS = 250             # Turning speed (was 500)
        self.RAMP_STEP_QPPS = 50         # Speed increment for ramping (was 100)
        self.RAMP_STEP_DELAY_S = 0.05    # Delay between speed steps
        
        # Obstacle avoidance
        self.OBSTACLE_METERS = 0.5       # Stop distance
        self.SLOW_METERS = 1.0           # Slow down distance
        
        # Person following
        self.PERSON_FOLLOW_DISTANCE = 0.5  # Target following distance
        self.SEARCH_TURN_TIME = 3.0        # Time to search when person lost
        
        # Movement smoothing
        self.current_left_speed = 0
        self.current_right_speed = 0
        self.target_left_speed = 0
        self.target_right_speed = 0
        
        # YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        self.confidence_threshold = 0.5
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # State tracking
        self.person_detected = False
        self.last_person_time = 0
        self.searching = False
        self.search_start_time = 0
        
    def setup(self):
        """Initialize robot systems"""
        print("Setting up Human Following Robot...")
        
        # Connect to RoboClaw
        if not self.connection_manager.wait_for_connection():
            print("Failed to connect to RoboClaw")
            return False
        
        # Start heartbeat monitoring
        self.connection_manager.start_heartbeat()
        
        # Start RealSense
        try:
            self.pipeline.start(self.config)
            print("RealSense camera started")
        except Exception as e:
            print(f"Failed to start RealSense: {e}")
            return False
        
        print("Robot setup complete!")
        return True
    
    def safe_close_roboclaw(self, roboclaw_obj):
        """Safely close RoboClaw connection"""
        try:
            if hasattr(roboclaw_obj, '_port') and roboclaw_obj._port:
                roboclaw_obj._port.close()
        except Exception as e:
            print(f"Error closing RoboClaw: {e}")
    
    def set_motor_speeds(self, left_speed, right_speed):
        """Set motor speeds with safety checks"""
        if not self.connection_manager.ensure_connection():
            print("Cannot set motor speeds - no connection")
            return False
        
        roboclaw = self.connection_manager.get_roboclaw()
        if not roboclaw:
            print("No RoboClaw instance available")
            return False
        
        try:
            # M1: left motor, M2: right motor (inverted)
            roboclaw.SpeedM1(self.connection_manager.address, left_speed)
            roboclaw.SpeedM2(self.connection_manager.address, -right_speed)  # M2 inverted
            return True
        except Exception as e:
            print(f"Error setting motor speeds: {e}")
            return False
    
    def stop_motors(self):
        """Stop both motors smoothly"""
        print("Stopping motors smoothly")
        self.ramp_to_speed(0, 0, duration=0.3)
        self.current_left_speed = 0
        self.current_right_speed = 0
    
    def ramp_to_speed(self, target_left, target_right, duration=0.5):
        """Smoothly ramp to target speeds, always through zero for direction changes"""
        if target_left == self.current_left_speed and target_right == self.current_right_speed:
            return
        
        # Check if we need to change direction (forward to backward or vice versa)
        direction_change = False
        if (self.current_left_speed > 0 and target_left < 0) or (self.current_left_speed < 0 and target_left > 0):
            direction_change = True
            print(f"Direction change detected - ramping through zero")
        
        if direction_change:
            # Always ramp through zero for direction changes
            # Phase 1: Ramp to zero
            print(f"Phase 1: Ramping to zero")
            self.ramp_to_speed(0, 0, duration/2)
            
            # Phase 2: Ramp from zero to target
            print(f"Phase 2: Ramping from zero to target")
            self.ramp_to_speed(target_left, target_right, duration/2)
        else:
            # Normal ramping (same direction)
            steps = max(1, int(duration / self.RAMP_STEP_DELAY_S))
            left_step = (target_left - self.current_left_speed) / steps
            right_step = (target_right - self.current_right_speed) / steps
            
            for i in range(steps):
                left_speed = int(self.current_left_speed + left_step * (i + 1))
                right_speed = int(self.current_right_speed + right_step * (i + 1))
                
                if self.set_motor_speeds(left_speed, right_speed):
                    self.current_left_speed = left_speed
                    self.current_right_speed = right_speed
                
                time.sleep(self.RAMP_STEP_DELAY_S)
    
    def get_center_distance(self, depth_frame):
        """Get distance at image center"""
        if not depth_frame:
            return float('inf')
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        distance = depth_frame.get_distance(width // 2, height // 2)
        return distance if distance > 0 else float('inf')
    
    def detect_person(self, color_frame, depth_frame):
        """Detect person using YOLO and return position/distance"""
        if not color_frame or not depth_frame:
            return None
        
        # Convert to numpy array for YOLO
        color_image = np.asanyarray(color_frame.get_data())
        
        # Run YOLO detection
        results = self.yolo_model(color_image, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0)
                    if int(box.cls[0]) == 0 and box.conf[0] > self.confidence_threshold:
                        # Get bounding box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Calculate distance at person center
                        distance = depth_frame.get_distance(int(center_x), int(center_y))
                        if distance > 0:
                            return {
                                'center_x': center_x,
                                'center_y': center_y,
                                'distance': distance,
                                'confidence': float(box.conf[0])
                            }
        
        return None
    
    def calculate_movement(self, person_info, center_distance):
        """Calculate movement based on person position and obstacles"""
        left_speed = 0
        right_speed = 0
        
        # Priority 1: Obstacle avoidance
        if center_distance <= self.OBSTACLE_METERS:
            print(f"OBSTACLE DETECTED: {center_distance:.2f}m - STOPPING")
            return 0, 0
        
        # Priority 2: Person following
        if person_info:
            person_x = person_info['center_x']
            person_distance = person_info['distance']
            
            # Calculate lateral error (how far person is from center)
            image_center = 320  # 640/2
            lateral_error = person_x - image_center
            
            print(f"PERSON DETECTED:")
            print(f"  Distance: {person_distance:.2f}m (target: {self.PERSON_FOLLOW_DISTANCE:.2f}m)")
            print(f"  Position: {person_x:.1f}px from center (error: {lateral_error:.1f}px)")
            
            # Distance control
            if person_distance < self.PERSON_FOLLOW_DISTANCE - 0.1:
                # Too close - back away slowly
                base_speed = -self.FORWARD_TARGET_QPPS // 4
                print(f"  Too close - backing away slowly")
            elif person_distance > self.PERSON_FOLLOW_DISTANCE + 0.1:
                # Too far - move forward
                base_speed = self.FORWARD_TARGET_QPPS
                print(f"  Too far - moving forward to catch up")
            else:
                # Good distance - maintain speed
                base_speed = self.FORWARD_TARGET_QPPS // 2
                print(f"  Good distance - maintaining speed")
            
            # Turning control
            if abs(lateral_error) > 100:  # Increased tolerance - person can be anywhere in center 200px wide zone
                if lateral_error > 0:
                    # Person to the right - turn right
                    turn_speed = self.TURN_QPPS
                    print(f"  Person to right - turning right while moving")
                    left_speed = base_speed + turn_speed
                    right_speed = base_speed - turn_speed
                else:
                    # Person to the left - turn left
                    turn_speed = self.TURN_QPPS
                    print(f"  Person to left - turning left while moving")
                    left_speed = base_speed - turn_speed
                    right_speed = base_speed + turn_speed
            else:
                # Person centered - move straight (within 100px tolerance)
                print(f"  Person centered - moving straight (error: {lateral_error:.1f}px)")
                left_speed = base_speed
                right_speed = base_speed
            
            # Ensure speeds are within limits
            left_speed = max(-self.FORWARD_TARGET_QPPS, min(self.FORWARD_TARGET_QPPS, left_speed))
            right_speed = max(-self.FORWARD_TARGET_QPPS, min(self.FORWARD_TARGET_QPPS, right_speed))
            
            print(f"  Final speeds: Left={left_speed}, Right={right_speed}")
            
        else:
            # No person detected - search behavior
            if not self.searching:
                self.searching = True
                self.search_start_time = time.time()
                print("No person detected - starting search pattern")
            
            search_elapsed = time.time() - self.search_start_time
            if search_elapsed < self.SEARCH_TURN_TIME:
                # Turn in place to search
                left_speed = -self.TURN_QPPS // 2
                right_speed = -self.TURN_QPPS // 2
                print(f"Searching: turning left ({search_elapsed:.1f}s elapsed)")
            else:
                # Stop searching
                self.searching = False
                print("Search timeout - stopping")
                left_speed = 0
                right_speed = 0
        
        return left_speed, right_speed
    
    def run(self):
        """Main robot control loop"""
        print("Starting robot control loop...")
        
        try:
            while True:
                # Get camera frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print("No camera frames available")
                    time.sleep(0.1)
                    continue
                
                # Check center distance for obstacles
                center_distance = self.get_center_distance(depth_frame)
                
                # Detect person
                person_info = self.detect_person(color_frame, depth_frame)
                
                # Update person detection state
                if person_info:
                    self.person_detected = True
                    self.last_person_time = time.time()
                    self.searching = False
                else:
                    # Person lost if not seen for 2 seconds
                    if time.time() - self.last_person_time > 2.0:
                        self.person_detected = False
                
                # Calculate movement
                left_speed, right_speed = self.calculate_movement(person_info, center_distance)
                
                # Apply movement with smoothing
                if left_speed != self.current_left_speed or right_speed != self.current_right_speed:
                    print(f"MOVEMENT: Left={left_speed}, Right={right_speed}")
                    # Use longer duration for smoother movement, especially for direction changes
                    ramp_duration = 0.3 if (left_speed * self.current_left_speed < 0) else 0.2
                    self.ramp_to_speed(left_speed, right_speed, duration=ramp_duration)
                
                # Small delay for control loop
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Stopping robot...")
        except Exception as e:
            print(f"Error in control loop: {e}")
        finally:
            self.stop_motors()
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.stop_motors()
        self.connection_manager.close()
        self.pipeline.stop()
        print("Cleanup complete")

def main():
    """Main function"""
    robot = HumanFollowingRobot()
    
    if robot.setup():
        try:
            robot.run()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            robot.cleanup()
    else:
        print("Failed to setup robot")

if __name__ == "__main__":
    main()
