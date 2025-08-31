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

class PeopleFollowingRobot:
    """People following robot with obstacle avoidance"""
    
    def __init__(self):
        # RoboClaw connection
        self.connection_manager = RoboClawConnectionManager()
        
        # Motor control parameters - all speeds at least 150 for better performance
        self.FORWARD_SPEED = 400         # Base forward speed (doubled from 200)
        self.TURN_SPEED = 150            # Base turning speed (at least 150)
        self.PERSON_CENTERING_SPEED = 80 # Reduced speed specifically for person centering
        self.BACKUP_SPEED = 300          # Backup speed (doubled from 150)
        self.SEARCH_SPEED = 150          # Increased search speed (at least 150)
        self.RAMP_STEP_DELAY_S = 0.05    # Delay between speed steps
        
        # Obstacle avoidance
        self.OBSTACLE_METERS = 0.3       # Stop distance for obstacles (reduced from 0.5)
        self.SLOW_METERS = 1.5           # Slow down distance (increased from 1.0)
        
        # Person following
        self.PERSON_STOP_DISTANCE = 0.5   # Stop distance from person
        self.PERSON_SLOW_DISTANCE = 1.0   # Slow down distance from person
        self.PERSON_BACKUP_DISTANCE = 0.5 # Start backing up when closer than this
        
        # Movement smoothing
        self.current_left_speed = 0
        self.current_right_speed = 0
        
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
        self.avoiding_obstacle = False
        self.last_person_x = 320  # Track last known person position for recovery
        self.person_exit_direction = 0  # -1 for left, 0 for center, 1 for right
        
        # Position smoothing for stable person tracking
        self.person_x_history = [320] * 5  # Store last 5 person positions
        self.smoothed_person_x = 320  # Smoothed person position
        
    def setup(self):
        """Initialize robot systems"""
        print("Setting up People Following Robot...")
        
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
            # M1: left motor (positive), M2: right motor (negative) for forward movement
            roboclaw.SpeedM1(self.connection_manager.address, left_speed)
            roboclaw.SpeedM2(self.connection_manager.address, -right_speed)
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
    
    def calculate_ramp_duration(self, target_left, target_right):
        """Calculate optimal ramp duration based on speed change magnitude"""
        # Calculate total speed change
        left_change = abs(target_left - self.current_left_speed)
        right_change = abs(target_right - self.current_right_speed)
        total_change = left_change + right_change
        
        # Check for direction changes
        direction_change = False
        if (self.current_left_speed > 0 and target_left < 0) or (self.current_left_speed < 0 and target_left > 0):
            direction_change = True
        if (self.current_right_speed > 0 and target_right < 0) or (self.current_right_speed < 0 and target_right > 0):
            direction_change = True
        
        if direction_change:
            # Direction changes always need longer ramps for safety
            return 0.3
        elif total_change < 30:  # Very small adjustment
            return 0.02  # Almost instant
        elif total_change < 80:  # Small adjustment
            return 0.05  # Very fast ramp
        elif total_change < 150:  # Medium adjustment
            return 0.1   # Fast ramp
        else:  # Large change
            return 0.2   # Moderate ramp
    
    def ramp_to_speed(self, target_left, target_right, duration=None):
        """Smoothly ramp to target speeds with adaptive duration"""
        if target_left == self.current_left_speed and target_right == self.current_right_speed:
            return
        
        # Calculate optimal ramp duration if not specified
        if duration is None:
            duration = self.calculate_ramp_duration(target_left, target_right)
        
        # Check if we need to change direction
        direction_change = False
        if (self.current_left_speed > 0 and target_left < 0) or (self.current_left_speed < 0 and target_left > 0):
            direction_change = True
        if (self.current_right_speed > 0 and target_right < 0) or (self.current_right_speed < 0 and target_right > 0):
            direction_change = True
        
        if direction_change:
            # Always ramp through zero for direction changes
            print(f"Direction change detected - ramping through zero")
            # Phase 1: Ramp to zero
            self.ramp_to_speed(0, 0, duration/2)
            # Phase 2: Ramp from zero to target
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
    
    def get_obstacle_distance(self, depth_frame):
        """Scan 31x21 pixel grid around center for obstacles (increased from 21x11)"""
        if not depth_frame:
            return float('inf')
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        
        # Scan 31x21 grid around center (increased coverage)
        min_distance = float('inf')
        for i in range(31):
            for j in range(21):
                x = (width // 2) - 15 + i
                y = (height // 2) - 10 + j
                distance = depth_frame.get_distance(x, y)
                if distance > 0:
                    min_distance = min(distance, min_distance)
        
        return min_distance
    
    def smooth_person_position(self, new_x):
        """Smooth person position using moving average to reduce jitter"""
        # Add new position to history
        self.person_x_history.append(new_x)
        # Keep only last 5 positions
        if len(self.person_x_history) > 5:
            self.person_x_history.pop(0)
        
        # Calculate smoothed position (simple moving average)
        self.smoothed_person_x = sum(self.person_x_history) / len(self.person_x_history)
        return self.smoothed_person_x

    def detect_person(self, color_frame, depth_frame):
        """Detect person using YOLO and return position/distance"""
        if not color_frame or not depth_frame:
            return None
        
        # Convert to numpy array for YOLO
        color_image = np.asanyarray(color_frame.get_data())
        
        # Run YOLO detection
        results = self.yolo_model(color_image, verbose=False)
        
        closest_person = None
        min_distance = float('inf')
        
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
                        if distance > 0 and distance < min_distance:
                            min_distance = distance
                            # Smooth the person position to reduce jitter
                            smoothed_center_x = self.smooth_person_position(center_x)
                            closest_person = {
                                'center_x': smoothed_center_x,
                                'center_y': center_y,
                                'distance': distance,
                                'confidence': float(box.conf[0])
                            }
        
        return closest_person
    
    def calculate_movement(self, person_info, obstacle_distance):
        """Calculate movement based on person position and obstacles"""
        left_speed = 0
        right_speed = 0
        
        # Priority 1: Obstacle avoidance (non-person obstacles)
        if obstacle_distance <= self.OBSTACLE_METERS:
            if not self.avoiding_obstacle:
                print(f"OBSTACLE DETECTED: {obstacle_distance:.2f}m - STARTING OBSTACLE AVOIDANCE")
                self.avoiding_obstacle = True
            
            # If very close (within 0.3m), backup first then turn
            if obstacle_distance <= 0.3:
                print(f"  Very close obstacle: backing up first (distance: {obstacle_distance:.2f}m)")
                left_speed = -self.BACKUP_SPEED
                right_speed = -self.BACKUP_SPEED
                return left_speed, right_speed
            else:
                # Turn left until obstacle is clear (at least 150 speed)
                left_speed = -max(150, self.TURN_SPEED // 2)
                right_speed = -max(150, self.TURN_SPEED // 2)
                print(f"  Avoiding obstacle: turning left (distance: {obstacle_distance:.2f}m)")
                return left_speed, right_speed
        elif self.avoiding_obstacle:
            # Obstacle cleared, resume normal operation
            print(f"OBSTACLE CLEARED: {obstacle_distance:.2f}m - RESUMING NORMAL OPERATION")
            self.avoiding_obstacle = False
        
        # Priority 2: Person following
        if person_info:
            person_x = person_info['center_x']
            person_distance = person_info['distance']
            
            # Calculate lateral error (how far person is from center)
            image_center = 320  # 640/2
            lateral_error = person_x - image_center
            
            print(f"PERSON DETECTED:")
            print(f"  Distance: {person_distance:.2f}m")
            print(f"  Position: {person_x:.1f}px from center (error: {lateral_error:.1f}px)")
            
            # Distance control - less aggressive speed reduction for faster movement
            if person_distance < self.PERSON_BACKUP_DISTANCE:
                # Too close - back away at moderate speed
                base_speed = -self.BACKUP_SPEED // 2  # Reduced backup speed
                print(f"  Too close - backing away at moderate speed")
            elif person_distance < self.PERSON_STOP_DISTANCE:
                # Very close - slow forward instead of stopping
                base_speed = self.FORWARD_SPEED // 4  # Very slow forward
                print(f"  Very close - moving very slowly forward")
            elif person_distance < self.PERSON_SLOW_DISTANCE:
                # Close - moderate speed instead of half speed
                base_speed = self.FORWARD_SPEED * 3 // 4  # 75% of full speed
                print(f"  Close - moving at moderate speed")
            else:
                # Good distance - full speed
                base_speed = self.FORWARD_SPEED
                print(f"  Good distance - full speed")
            
            # Turning control - turn toward person to keep them centered
            if abs(lateral_error) > 200:  # Person can be anywhere in center 400px wide zone (increased from 300px) (increased tolerance)
                if lateral_error > 0:
                    # Person to the right - turn right toward them (slower for stability)
                    turn_speed = self.PERSON_CENTERING_SPEED
                    print(f"  Person to right - turning right toward them (slow and stable)")
                    left_speed = base_speed + turn_speed
                    right_speed = base_speed - turn_speed
                else:
                    # Person to the left - turn left toward them (slower for stability)
                    turn_speed = self.PERSON_CENTERING_SPEED
                    print(f"  Person to left - turning left toward them (slow and stable)")
                    left_speed = base_speed - turn_speed
                    right_speed = base_speed + turn_speed
            else:
                # Person centered - move straight
                print(f"  Person centered - moving straight")
                left_speed = base_speed
                right_speed = base_speed
            
            # Ensure speeds are within limits
            max_speed = self.FORWARD_SPEED * 2
            left_speed = max(-max_speed, min(max_speed, left_speed))
            right_speed = max(-max_speed, min(max_speed, right_speed))
            
            print(f"  Final speeds: Left={left_speed}, Right={right_speed}")
            
        else:
            # No person detected - smart recovery mode based on exit direction
            if not self.searching:
                self.searching = True
                print("No person detected - entering smart recovery mode")
            
            # Smart recovery: turn toward where person left with speeds at least 150
            if self.person_exit_direction == -1:  # Person left to the left
                print("Recovery: turning LEFT toward where person left (speed at least 150)")
                left_speed = -max(150, self.SEARCH_SPEED)  # Left turn at least 150
                right_speed = max(150, self.SEARCH_SPEED)   # Right wheel forward at least 150
            elif self.person_exit_direction == 1:  # Person left to the right
                print("Recovery: turning RIGHT toward where person left (speed at least 150)")
                left_speed = max(150, self.SEARCH_SPEED)    # Left wheel forward at least 150
                right_speed = -max(150, self.SEARCH_SPEED)  # Right turn at least 150
            else:  # Person left from center or unknown
                print("Recovery: searching in center area with speeds at least 150")
                left_speed = max(150, self.FORWARD_SPEED - self.SEARCH_SPEED)
                right_speed = max(150, self.FORWARD_SPEED + self.SEARCH_SPEED)
        
        return left_speed, right_speed
    
    def draw_obstacle_zone(self, image, depth_frame):
        """Draw the 31x21 obstacle detection grid on the image (increased from 21x11)"""
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw the scanning grid (31x21)
        for i in range(31):
            for j in range(21):
                x = center_x - 15 + i
                y = center_y - 10 + j
                
                if 0 <= x < width and 0 <= y < height:
                    # Get distance at this point
                    distance = depth_frame.get_distance(x, y)
                    if distance > 0:
                        # Color code based on distance
                        if distance <= self.OBSTACLE_METERS:
                            color = (0, 0, 255)  # Red for close obstacles
                        elif distance <= self.SLOW_METERS:
                            color = (0, 255, 255)  # Yellow for slow zone
                        else:
                            color = (0, 255, 0)  # Green for safe distance
                        
                        cv2.circle(image, (x, y), 2, color, -1)
        
        # Draw center crosshair
        cv2.line(image, (center_x-15, center_y), (center_x+15, center_y), (255, 255, 255), 2)
        cv2.line(image, (center_x, center_y-15), (center_x, center_y+15), (255, 255, 255), 2)
    
    def draw_person_info(self, image, person_info):
        """Draw person detection information on the image"""
        x, y = int(person_info['center_x']), int(person_info['center_y'])
        distance = person_info['distance']
        confidence = person_info['confidence']
        
        # Draw bounding box (approximate)
        box_size = 50
        cv2.rectangle(image, (x-box_size//2, y-box_size//2), (x+box_size//2, y+box_size//2), (0, 255, 0), 2)
        
        # Draw center point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        # Draw distance and confidence text
        text = f"Person: {distance:.2f}m ({confidence:.2f})"
        cv2.putText(image, text, (x-box_size//2, y-box_size//2-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_status_info(self, image, obstacle_distance, person_info):
        """Draw status information on the image"""
        height, width = image.shape[:2]
        
        # Background for text
        cv2.rectangle(image, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Status text
        y_offset = 30
        cv2.putText(image, f"Obstacle Distance: {obstacle_distance:.2f}m", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        if person_info:
            cv2.putText(image, f"Person Detected: YES", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(image, f"Person Distance: {person_info['distance']:.2f}m", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(image, f"Person Detected: NO", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_offset += 25
        if self.avoiding_obstacle:
            cv2.putText(image, f"Status: AVOIDING OBSTACLE", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.searching:
            cv2.putText(image, f"Status: SEARCHING", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(image, f"Status: FOLLOWING", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def run(self):
        """Main robot control loop"""
        print("Starting people following robot...")
        print("Press 'q' to quit the viewer")
        
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
                
                # Convert frames to numpy arrays for display
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create display image
                display_image = color_image.copy()
                
                # Check for obstacles (non-person)
                obstacle_distance = self.get_obstacle_distance(depth_frame)
                
                # Detect person
                person_info = self.detect_person(color_frame, depth_frame)
                
                # Draw obstacle detection zone (31x21 grid)
                self.draw_obstacle_zone(display_image, depth_frame)
                
                # Draw person detection info
                if person_info:
                    self.draw_person_info(display_image, person_info)
                
                # Draw status information
                self.draw_status_info(display_image, obstacle_distance, person_info)
                
                # Update person detection state
                if person_info:
                    self.person_detected = True
                    self.last_person_time = time.time()
                    self.last_person_x = person_info['center_x']  # Track person position
                    self.searching = False
                else:
                    # Person lost if not seen for 2 seconds
                    if time.time() - self.last_person_time > 2.0:
                        if self.person_detected:  # Just lost person
                            # Determine exit direction based on last known position
                            image_center = 320
                            if self.last_person_x < image_center - 50:  # Left side
                                self.person_exit_direction = -1
                                print(f"Person left frame to the LEFT (last position: {self.last_person_x:.1f})")
                            elif self.last_person_x > image_center + 50:  # Right side
                                self.person_exit_direction = 1
                                print(f"Person left frame to the RIGHT (last position: {self.last_person_x:.1f})")
                            else:  # Center
                                self.person_exit_direction = 0
                                print(f"Person left frame from CENTER (last position: {self.last_person_x:.1f})")
                        self.person_detected = False
                
                # Calculate movement
                left_speed, right_speed = self.calculate_movement(person_info, obstacle_distance)
                
                # Apply movement with adaptive smoothing
                if left_speed != self.current_left_speed or right_speed != self.current_right_speed:
                    print(f"MOVEMENT: Left={left_speed}, Right={right_speed}")
                    # Use adaptive ramping - small changes are almost instant, large changes are smooth
                    self.ramp_to_speed(left_speed, right_speed)
                
                # Display the image
                cv2.imshow('Robot Vision - People Following & Obstacle Avoidance', display_image)
                
                # Check for key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quit requested by user")
                    break
                
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
    robot = PeopleFollowingRobot()
    
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
