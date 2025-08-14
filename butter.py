from roboclaw import Roboclaw
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import threading
import glob
import os
import numpy as np
from scipy.interpolate import CubicSpline
from collections import deque


@dataclass
class PersonInfo:
    """Information about a detected person with enhanced tracking"""
    distance: float  # meters from camera
    confidence: float  # YOLOv8 confidence (0.0 to 1.0)
    bbox: List[float]  # [x1, y1, x2, y2] coordinates
    center_x: float  # center of person in image
    center_y: float  # center of person in image
    velocity_x: float = 0.0  # Person's horizontal velocity (pixels/second)
    velocity_y: float = 0.0  # Person's vertical velocity (pixels/second)
    tracking_id: int = -1  # Unique ID for consistent tracking
    last_seen: float = 0.0  # Timestamp of last detection
    tracking_confidence: float = 0.0  # How confident we are in tracking
    depth_samples: List[float] = None  # Multiple depth readings for accuracy

@dataclass
class PersonHistory:
    """Historical data for person tracking"""
    positions: deque  # Recent position history
    velocities: deque  # Recent velocity history
    timestamps: deque  # Timestamps for each measurement
    kalman_filter: any = None  # Kalman filter for this person
    max_age: float = 2.0  # Maximum age of tracking data

@dataclass
class ObstacleInfo:
    """Information about detected obstacles"""
    distance: float  # meters from camera
    type: str  # object type
    confidence: float  # YOLOv8 confidence
    bbox: List[float]  # [x1, y1, x2, y2] coordinates

class RoboClawConnectionManager:
    """Manages RoboClaw connection with heartbeat monitoring and automatic reconnection"""
    
    def __init__(self, port="/dev/ttyACM0", baudrate=38400, address=0x80, heartbeat_interval=1.0):
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.heartbeat_interval = heartbeat_interval
        self.roboclaw = None
        self.connected = False
        self.connection_lock = threading.Lock()
        self.heartbeat_thread = None
        self.stop_heartbeat = False
        
    def check_port_permissions(self, port):
        """Check if we have proper permissions to access the port"""
        try:
            if not os.path.exists(port):
                print(f"  ✗ Port {port} does not exist")
                return False
            
            # Check if we can read the port
            if not os.access(port, os.R_OK):
                print(f"  ✗ No read permission on {port}")
                return False
            
            # Check if we can write to the port
            if not os.access(port, os.W_OK):
                print(f"  ✗ No write permission on {port}")
                return False
            
            print(f"  ✓ Port {port} permissions OK")
            return True
            
        except Exception as e:
            print(f"  ✗ Error checking permissions on {port}: {e}")
            return False
    
    def find_roboclaw_ports(self):
        """Scan for available RoboClaw devices on common USB ports"""
        possible_ports = []
        
        # Only check ttyACM0, ttyACM1, and ttyACM2 as requested
        port_patterns = [
            "/dev/ttyACM0",
            "/dev/ttyACM1",
            "/dev/ttyACM2",
        ]
        
        for port in port_patterns:
            if os.path.exists(port):
                possible_ports.append(port)
        
        return possible_ports
    
    def safe_close_roboclaw(self, roboclaw_obj):
        """Safely close a RoboClaw connection, handling various edge cases"""
        if roboclaw_obj is None:
            return
            
        try:
            # Try to close the underlying serial port
            if hasattr(roboclaw_obj, '_port') and roboclaw_obj._port is not None:
                roboclaw_obj._port.close()
                print("  ✓ RoboClaw connection closed successfully")
            else:
                print("  ⚠ No serial port to close")
        except Exception as e:
            print(f"  ⚠ Warning: Could not close RoboClaw connection: {e}")
            # Fallback: try to delete the object to trigger garbage collection
            try:
                del roboclaw_obj
            except:
                pass
    
    def test_roboclaw_connection(self, port):
        """Test if a specific port has a RoboClaw device"""
        test_roboclaw = None
        try:
            print(f"  Opening connection to {port}...")
            
            # Check permissions first
            if not self.check_port_permissions(port):
                return False, None, None
            
            test_roboclaw = Roboclaw(port, self.baudrate)
            test_roboclaw.Open()
            
            print(f"  Testing RoboClaw communication on {port}...")
            
            # Only try address 0x80 as requested
            addr = 0x80
            try:
                print(f"    Trying address 0x{addr:02X}...")
                result, version = test_roboclaw.ReadVersion(addr)
                
                if result:
                    print(f"  ✓ Successfully communicated with RoboClaw on {port} at address 0x{addr:02X}")
                    print(f"  ✓ RoboClaw version: {version}")
                    self.safe_close_roboclaw(test_roboclaw)
                    return True, version, addr
                else:
                    print(f"    ✗ No response from address 0x{addr:02X}")
                    
            except Exception as e:
                print(f"    ✗ Exception on address 0x{addr:02X}: {e}")
            
            print(f"  ✗ No RoboClaw found on {port}")
            self.safe_close_roboclaw(test_roboclaw)
            return False, None, None
                
        except Exception as e:
            print(f"  ✗ Exception on {port}: {e}")
            self.safe_close_roboclaw(test_roboclaw)
            return False, None, None
    
    def scan_and_connect(self):
        """Scan all available ports and connect to the first working RoboClaw"""
        print("Scanning for RoboClaw devices...")
        available_ports = self.find_roboclaw_ports()
        
        if not available_ports:
            print("No USB serial ports found")
            return False
        
        print(f"Found {len(available_ports)} potential ports: {available_ports}")
        
        # Try each port
        for port in available_ports:
            print(f"Testing port: {port}")
            is_roboclaw, version, discovered_address = self.test_roboclaw_connection(port)
            
            if is_roboclaw:
                # Found a working RoboClaw, connect to it
                try:
                    print(f"Attempting to establish connection to {port} at address 0x{discovered_address:02X}...")
                    self.roboclaw = Roboclaw(port, self.baudrate)
                    self.roboclaw.Open()
                    
                    # Update address if we discovered a different one
                    if discovered_address != self.address:
                        print(f"Updating RoboClaw address from 0x{self.address:02X} to 0x{discovered_address:02X}")
                        self.address = discovered_address
                    
                    # Verify connection
                    result, version = self.roboclaw.ReadVersion(self.address)
                    if result:
                        self.port = port  # Update the port to the working one
                        self.connected = True
                        print(f"Successfully connected to RoboClaw on {port} at address 0x{self.address:02X} with version: {version}")
                        return True
                    else:
                        print(f"Failed to verify connection on {port}")
                        self.safe_close_roboclaw(self.roboclaw)
                        self.roboclaw = None
                        
                except Exception as e:
                    print(f"Failed to connect to {port}: {e}")
                    if self.roboclaw:
                        self.safe_close_roboclaw(self.roboclaw)
                        self.roboclaw = None
        
        print("No working RoboClaw found on any port")
        return False
    
    def wait_for_connection(self, max_retries=None, retry_delay=2.0):
        """Wait for RoboClaw to connect, with optional retry limit"""
        retry_count = 0
        while max_retries is None or retry_count < max_retries:
            try:
                print(f"Attempting to connect to RoboClaw...")
                
                # First try the original port
                if os.path.exists(self.port):
                    print(f"Trying original port: {self.port}")
                    self.roboclaw = Roboclaw(self.port, self.baudrate)
                    self.roboclaw.Open()
                    
                    # Test connection by reading version
                    result, version = self.roboclaw.ReadVersion(self.address)
                    if result:
                        self.connected = True
                        print(f"RoboClaw connected successfully on {self.port}! Version: {version}")
                        self.start_heartbeat()
                        return True
                    else:
                        print("Failed to read RoboClaw version on original port, trying other ports...")
                        self.safe_close_roboclaw(self.roboclaw)
                        self.roboclaw = None
                
                # If original port failed, scan all ports
                if self.scan_and_connect():
                    self.start_heartbeat()
                    return True
                    
            except Exception as e:
                print(f"Connection attempt failed: {e}")
            
            retry_count += 1
            if max_retries is None or retry_count < max_retries:
                print(f"Retrying in {retry_delay} seconds... (attempt {retry_count})")
                time.sleep(retry_delay)
        
        print(f"Failed to connect after {retry_count} attempts")
        return False
    
    def start_heartbeat(self):
        """Start heartbeat monitoring in a separate thread"""
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.stop_heartbeat = False
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self.heartbeat_thread.start()
    
    def stop_heartbeat_monitoring(self):
        """Stop heartbeat monitoring"""
        self.stop_heartbeat = True
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2.0)
    
    def _heartbeat_loop(self):
        """Heartbeat monitoring loop"""
        while not self.stop_heartbeat:
            try:
                with self.connection_lock:
                    if self.roboclaw and self.connected:
                        # Try to read version as heartbeat
                        result, _ = self.roboclaw.ReadVersion(self.address)
                        if not result:
                            print("RoboClaw heartbeat failed - connection lost!")
                            self.connected = False
                            self.safe_close_roboclaw(self.roboclaw)
                            self.roboclaw = None
                            
            except Exception as e:
                print(f"Heartbeat error: {e}")
                with self.connection_lock:
                    self.connected = False
                    if self.roboclaw:
                        self.safe_close_roboclaw(self.roboclaw)
                        self.roboclaw = None
            
            time.sleep(self.heartbeat_interval)
    
    def ensure_connection(self):
        """Ensure RoboClaw is connected, reconnect if necessary"""
        with self.connection_lock:
            if not self.connected:
                print("RoboClaw disconnected, attempting to reconnect...")
                return self.wait_for_connection()
            return True
    
    def is_connected(self):
        """Check if RoboClaw is currently connected"""
        with self.connection_lock:
            return self.connected
    
    def get_roboclaw(self):
        """Get the RoboClaw instance if connected"""
        with self.connection_lock:
            return self.roboclaw if self.connected else None
    
    def close(self):
        """Close the connection and cleanup"""
        self.stop_heartbeat_monitoring()
        with self.connection_lock:
            if self.roboclaw:
                self.safe_close_roboclaw(self.roboclaw)
                self.roboclaw = None
            self.connected = False

class HumanFollowingRobot:
    def __init__(self):
        # Motor controller setup with connection manager
        self.connection_manager = RoboClawConnectionManager()
        self.address = 0x80
        
        # Wait for initial connection
        if not self.connection_manager.wait_for_connection():
            raise RuntimeError("Failed to establish initial RoboClaw connection")
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15) # 15 frames, maybe could go back to 30
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        self.pipeline.start(config)
        
        # YOLOv8 setup
        self.yolo_model_primary = YOLO('yolov8m.pt')  # Better accuracy
        self.yolo_model_backup = YOLO('yolov8n.pt')   # Faster fallback
        self.confidence_threshold = 0.75  # Higher confidence requirement
        
        # Enhanced tracking parameters
        self.tracking_id_counter = 0
        self.person_tracking_buffer = {}  # Dict of tracking_id -> PersonHistory
        self.max_tracking_age = 2.0      # How long to remember a person
        self.tracking_distance_threshold = 50  # pixels - max distance for same person
        self.velocity_history_length = 10  # How many velocity measurements to keep
        
        # Kalman filter setup for position smoothing
        self.kalman_filters = {}  # tracking_id -> KalmanFilter
        
        # Multi-point depth sampling
        self.depth_sample_points = [
            (0.5, 0.3),  # Head (relative to bbox)
            (0.5, 0.6),  # Chest
            (0.5, 0.9),  # Feet
        ]
        self.depth_weights = [0.4, 0.4, 0.2]  # Head, chest, feet weights
        
        # Control parameters (following avoid_obstacle.py conventions)
        self.RAMP_STEP_QPPS = 20  # Smaller steps for smoother ramping
        self.FORWARD_TARGET_QPPS = 50 * 40  # ~"full speed" target (doubled from 20)
        self.TURN_QPPS = 50 * 20  # turning speed magnitude (doubled from 10)
        self.RAMP_STEP_DELAY_S = 0.02  # Faster updates for smoother movement
        self.OBSTACLE_METERS = 0.5  # minimum safe distance (reduced from 1.0)
        self.PERSON_FOLLOW_DISTANCE = 1.0  # target following distance
        self.CHECK_INTERVAL_S = 0.05
        
        # Smooth movement parameters
        self.MICRO_ADJUSTMENT_THRESHOLD = 20  # pixels - when to make micro-adjustments
        self.MICRO_TURN_QPPS = 50 * 3  # very small turning speed for micro-adjustments
        self.SMOOTHING_FACTOR = 0.3  # how much to smooth speed changes (0-1)
        self.MAX_ACCELERATION = 50 * 5  # maximum speed change per update
        
        # Advanced interpolation parameters
        self.TRAJECTORY_POINTS = 50  # number of points in trajectory
        self.TRAJECTORY_TIME = 2.0  # seconds to plan ahead
        self.S_CURVE_ACCEL = True  # use S-curve acceleration profiles
        self.CUBIC_SPLINE_SMOOTHING = True  # use cubic splines for path smoothing
        self.PREDICTION_HORIZON = 0.5  # seconds to predict person movement
        
        # Trajectory buffers
        self.position_history = deque(maxlen=100)  # recent position history
        self.velocity_history = deque(maxlen=100)  # recent velocity history
        self.trajectory_buffer = deque(maxlen=self.TRAJECTORY_POINTS)  # planned trajectory
        self.last_trajectory_update = 0
        
        # Interpolation state
        self.current_trajectory = None
        self.trajectory_start_time = 0
        self.interpolation_active = False
        
        # State variables
        self.current_speed_m1 = 0
        self.current_speed_m2 = 0
        self.target_speed_m1 = 0
        self.target_speed_m2 = 0
        self.person_detected = False
        self.last_person_center_x = 320  # center of 640x480 image
        self.last_person_center_y = 240
        self.smooth_person_x = 320  # smoothed person position
        self.smooth_person_y = 240
        
        print("Human Following Robot initialized successfully")
    
    def get_center_distance(self, depth_frame: rs.depth_frame) -> float:
        """Return distance at image center in meters. Returns float('inf') if invalid."""
        if not depth_frame:
            return float("inf")
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        distance_meters = depth_frame.get_distance(width // 2, height // 2)
        return distance_meters if distance_meters > 0 else float("inf")
    
    def get_multi_point_depth(self, person_bbox: List[float], depth_frame) -> Tuple[float, List[float]]:
        """Get depth at multiple points on person for more accurate distance measurement"""
        if not depth_frame:
            return float("inf"), []
        
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = person_bbox
        bbox_width = bbox_x2 - bbox_x1
        bbox_height = bbox_y2 - bbox_y1
        
        depth_samples = []
        total_weighted_depth = 0.0
        total_weight = 0.0
        
        for (rel_x, rel_y), weight in zip(self.depth_sample_points, self.depth_weights):
            # Calculate absolute pixel coordinates
            pixel_x = int(bbox_x1 + rel_x * bbox_width)
            pixel_y = int(bbox_y1 + rel_y * bbox_height)
            
            # Ensure coordinates are within depth frame bounds
            pixel_x = max(0, min(pixel_x, depth_frame.get_width() - 1))
            pixel_y = max(0, min(pixel_y, depth_frame.get_height() - 1))
            
            # Get depth at this point
            depth = depth_frame.get_distance(pixel_x, pixel_y)
            
            if depth > 0:
                depth_samples.append(depth)
                total_weighted_depth += depth * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_average_depth = total_weighted_depth / total_weight
            return weighted_average_depth, depth_samples
        else:
            return float("inf"), []
    
    def calculate_person_velocity(self, person_id: int, current_pos: Tuple[float, float], current_time: float) -> Tuple[float, float]:
        """Calculate person's velocity based on position history"""
        if person_id not in self.person_tracking_buffer:
            return 0.0, 0.0
        
        history = self.person_tracking_buffer[person_id]
        
        # Add current position to history
        history.positions.append(current_pos)
        history.timestamps.append(current_time)
        
        # Keep only recent history
        while len(history.positions) > self.velocity_history_length:
            history.positions.popleft()
            history.timestamps.popleft()
        
        if len(history.positions) < 2:
            return 0.0, 0.0
        
        # Calculate velocity from recent positions
        recent_positions = list(history.positions)[-3:]  # Last 3 positions
        recent_timestamps = list(history.timestamps)[-3:]
        
        if len(recent_positions) >= 2:
            # Calculate velocity from last two positions
            pos1, pos2 = recent_positions[-2], recent_positions[-1]
            t1, t2 = recent_timestamps[-2], recent_timestamps[-1]
            
            if t2 > t1:
                dt = t2 - t1
                vx = (pos2[0] - pos1[0]) / dt
                vy = (pos2[1] - pos1[1]) / dt
                
                # Store velocity in history
                history.velocities.append((vx, vy))
                while len(history.velocities) > self.velocity_history_length:
                    history.velocities.popleft()
                
                return vx, vy
        
        return 0.0, 0.0
    
    def create_kalman_filter(self) -> any:
        """Create a new Kalman filter for person tracking"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
        
        # Measurement matrix: we measure x and y positions
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        
        # Transition matrix: predict next position based on current position and velocity
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], np.float32)
        
        # Process noise: how much we trust our model
        kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32) * 0.01
        
        # Measurement noise: how much we trust our measurements
        kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], np.float32) * 0.1
        
        return kalman
    
    def update_kalman_filter(self, person_id: int, measured_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Update Kalman filter and return filtered position"""
        if person_id not in self.kalman_filters:
            self.kalman_filters[person_id] = self.create_kalman_filter()
        
        kalman = self.kalman_filters[person_id]
        
        # Predict next state
        prediction = kalman.predict()
        
        # Update with measurement
        measurement = np.array([[measured_pos[0]], [measured_pos[1]]], np.float32)
        kalman.correct(measurement)
        
        # Return filtered position (first two elements are x, y)
        filtered_x = float(prediction[0])
        filtered_y = float(prediction[1])
        
        return filtered_x, filtered_y
    
    def assign_tracking_id(self, person_center: Tuple[float, float], current_time: float) -> int:
        """Assign tracking ID to person, either new or existing"""
        best_match_id = -1
        best_distance = float('inf')
        
        # Look for existing person to match
        for person_id, history in self.person_tracking_buffer.items():
            if current_time - history.timestamps[-1] < self.max_tracking_age:
                last_pos = history.positions[-1]
                distance = math.sqrt((person_center[0] - last_pos[0])**2 + (person_center[1] - last_pos[1])**2)
                
                if distance < best_distance and distance < self.tracking_distance_threshold:
                    best_distance = distance
                    best_match_id = person_id
        
        if best_match_id != -1:
            # Update existing person
            return best_match_id
        else:
            # Create new person
            new_id = self.tracking_id_counter
            self.tracking_id_counter += 1
            
            # Initialize history for new person
            self.person_tracking_buffer[new_id] = PersonHistory(
                positions=deque([person_center], maxlen=self.velocity_history_length),
                velocities=deque([(0.0, 0.0)], maxlen=self.velocity_history_length),
                timestamps=deque([current_time], maxlen=self.velocity_history_length),
                kalman_filter=self.create_kalman_filter()
            )
            
            return new_id
    
    def calculate_tracking_confidence(self, person_id: int, current_time: float) -> float:
        """Calculate tracking confidence based on detection consistency and age"""
        if person_id not in self.person_tracking_buffer:
            return 0.0
        
        history = self.person_tracking_buffer[person_id]
        
        # Base confidence on detection age
        age = current_time - history.timestamps[-1]
        age_confidence = max(0.0, 1.0 - (age / self.max_tracking_age))
        
        # Confidence based on position consistency
        if len(history.positions) >= 3:
            recent_positions = list(history.positions)[-3:]
            position_variance = self.calculate_position_variance(recent_positions)
            consistency_confidence = max(0.0, 1.0 - position_variance / 100.0)
        else:
            consistency_confidence = 0.5
        
        # Combine confidences
        total_confidence = (age_confidence * 0.6) + (consistency_confidence * 0.4)
        return min(1.0, max(0.0, total_confidence))
    
    def calculate_position_variance(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate variance in position measurements"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate centroid
        centroid_x = sum(pos[0] for pos in positions) / len(positions)
        centroid_y = sum(pos[1] for pos in positions) / len(positions)
        
        # Calculate variance
        variance = sum((pos[0] - centroid_x)**2 + (pos[1] - centroid_y)**2 for pos in positions) / len(positions)
        return variance
    
    def cleanup_old_tracking_data(self, current_time: float):
        """Remove old tracking data to prevent memory bloat"""
        person_ids_to_remove = []
        
        for person_id, history in self.person_tracking_buffer.items():
            if current_time - history.timestamps[-1] > self.max_tracking_age:
                person_ids_to_remove.append(person_id)
        
        for person_id in person_ids_to_remove:
            del self.person_tracking_buffer[person_id]
            if person_id in self.kalman_filters:
                del self.kalman_filters[person_id]
    
    def detect_people_and_obstacles(self, rgb_frame, depth_frame) -> Tuple[List[PersonInfo], List[ObstacleInfo]]:
        """Detect people and obstacles using enhanced YOLOv8 and depth camera"""
        people = []
        obstacles = []
        current_time = time.time()
        
        try:
            # Try primary model first, fallback to backup if needed
            try:
                results = self.yolo_model_primary(rgb_frame, verbose=False)
            except Exception as e:
                print(f"Primary model failed, using backup: {e}")
                results = self.yolo_model_backup(rgb_frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        object_type = result.names[int(box.cls)]
                        confidence = float(box.conf)
                        
                        # Only process high-confidence detections
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Calculate object center
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        
                        if object_type == "person":
                            # Get multi-point depth for more accurate distance
                            distance, depth_samples = self.get_multi_point_depth(bbox.tolist(), depth_frame)
                            
                            if distance > 0 and distance < float('inf'):
                                # Assign tracking ID and calculate velocity
                                tracking_id = self.assign_tracking_id((center_x, center_y), current_time)
                                
                                # Calculate velocity
                                velocity_x, velocity_y = self.calculate_person_velocity(tracking_id, (center_x, center_y), current_time)
                                
                                # Update Kalman filter for smooth position
                                filtered_x, filtered_y = self.update_kalman_filter(tracking_id, (center_x, center_y))
                                
                                # Calculate tracking confidence based on detection consistency
                                tracking_confidence = self.calculate_tracking_confidence(tracking_id, current_time)
                                
                                people.append(PersonInfo(
                                    distance=distance,
                                    confidence=confidence,
                                    bbox=bbox.tolist(),
                                    center_x=filtered_x,  # Use Kalman-filtered position
                                    center_y=filtered_y,
                                    velocity_x=velocity_x,
                                    velocity_y=velocity_y,
                                    tracking_id=tracking_id,
                                    last_seen=current_time,
                                    tracking_confidence=tracking_confidence,
                                    depth_samples=depth_samples
                                ))
                        else:
                            # Handle obstacles with single-point depth
                            depth_x = int(center_x)
                            depth_y = int(center_y)
                            
                            # Ensure coordinates are within depth frame bounds
                            depth_x = max(0, min(depth_x, depth_frame.get_width() - 1))
                            depth_y = max(0, min(depth_y, depth_frame.get_height() - 1))
                            
                            distance = depth_frame.get_distance(depth_x, depth_y)
                            
                            if distance > 0:  # Valid depth reading
                                obstacles.append(ObstacleInfo(
                                    distance=distance,
                                    type=object_type,
                                    confidence=confidence,
                                    bbox=bbox.tolist()
                                ))
        
        except Exception as e:
            print(f"Error in enhanced YOLOv8 detection: {e}")
        
        # Clean up old tracking data
        self.cleanup_old_tracking_data(current_time)
        
        return people, obstacles
    
    def calculate_adaptive_distance(self, person: PersonInfo) -> float:
        """Calculate adaptive following distance based on person's behavior and environment"""
        base_distance = self.PERSON_FOLLOW_DISTANCE
        
        # Adjust based on person's walking speed
        person_speed = math.sqrt(person.velocity_x**2 + person.velocity_y**2)
        if person_speed > 50:  # Fast walking
            base_distance += 0.3  # Increase distance for safety
        elif person_speed < 10:  # Slow/stopped
            base_distance -= 0.2  # Decrease distance for closer following
        
        # Adjust based on tracking confidence
        if person.tracking_confidence < 0.7:
            base_distance += 0.2  # Increase distance if tracking is uncertain
        
        # Ensure minimum safe distance
        return max(0.5, min(2.0, base_distance))
    
    def match_person_velocity(self, person: PersonInfo) -> int:
        """Calculate robot speed to match person's walking speed"""
        person_speed = math.sqrt(person.velocity_x**2 + person.velocity_y**2)
        
        # Convert pixel velocity to approximate m/s (rough estimation)
        # Assuming 640x480 image and typical person size
        estimated_mps = person_speed * 0.01  # Rough conversion factor
        
        # Calculate target robot speed - ALWAYS POSITIVE for forward movement
        if estimated_mps > 0.5:  # Person is walking
            target_speed = int(self.FORWARD_TARGET_QPPS * min(estimated_mps / 1.0, 1.5))
        else:  # Person is stopped or slow
            target_speed = self.FORWARD_TARGET_QPPS // 4  # Very slow following
        
        # CRITICAL FIX: Ensure speed is ALWAYS POSITIVE for forward movement
        target_speed = abs(target_speed)  # Force positive
        
        # Ensure speed is within safe limits
        return max(100, min(self.FORWARD_TARGET_QPPS, target_speed))  # Minimum 100 QPPS
    
    def predict_person_path(self, person: PersonInfo, time_horizon: float) -> Tuple[float, float]:
        """Predict person's future position using velocity and acceleration analysis"""
        current_time = time.time()
        
        if person.tracking_id not in self.person_tracking_buffer:
            return person.center_x, person.center_y
        
        history = self.person_tracking_buffer[person.tracking_id]
        
        if len(history.velocities) < 2:
            return person.center_x, person.center_y
        
        # Get recent velocities
        recent_velocities = list(history.velocities)[-3:]
        
        # Calculate acceleration
        if len(recent_velocities) >= 2:
            v1 = recent_velocities[-2]
            v2 = recent_velocities[-1]
            dt = 0.1  # Assuming 10fps
            
            ax = (v2[0] - v1[0]) / dt
            ay = (v2[1] - v1[1]) / dt
            
            # Predict future position with acceleration
            predicted_x = person.center_x + person.velocity_x * time_horizon + 0.5 * ax * time_horizon**2
            predicted_y = person.center_y + person.velocity_y * time_horizon + 0.5 * ay * time_horizon**2
        else:
            # Simple velocity-based prediction
            predicted_x = person.center_x + person.velocity_x * time_horizon
            predicted_y = person.center_y + person.velocity_y * time_horizon
        
        return predicted_x, predicted_y
    
    def analyze_environment_context(self, person: PersonInfo, obstacles: List[ObstacleInfo]) -> dict:
        """Analyze environment context to adjust robot behavior"""
        context = {
            'person_walking_straight': False,
            'person_making_turn': False,
            'person_stopping': False,
            'person_changing_speed': False,
            'obstacle_density': 'low',
            'recommended_behavior': 'normal_following'
        }
        
        # Analyze person's movement pattern
        if person.tracking_id in self.person_tracking_buffer:
            history = self.person_tracking_buffer[person.tracking_id]
            
            if len(history.positions) >= 5:
                # Check if person is walking in a straight line
                recent_positions = list(history.positions)[-5:]
                straightness = self.calculate_path_straightness(recent_positions)
                context['person_walking_straight'] = straightness > 0.8
                
                # Check if person is making a turn
                context['person_making_turn'] = straightness < 0.6
                
                # Check if person is stopping
                recent_velocities = list(history.velocities)[-3:]
                if len(recent_velocities) >= 2:
                    speed_change = abs(recent_velocities[-1][0] + recent_velocities[-1][1]) - abs(recent_velocities[-2][0] + recent_velocities[-2][1])
                    context['person_stopping'] = speed_change < -20
                    context['person_changing_speed'] = abs(speed_change) > 10
        
        # Analyze obstacle density
        nearby_obstacles = [obs for obs in obstacles if obs.distance < 2.0]
        if len(nearby_obstacles) > 3:
            context['obstacle_density'] = 'high'
            context['recommended_behavior'] = 'cautious_following'
        elif len(nearby_obstacles) > 1:
            context['obstacle_density'] = 'medium'
            context['recommended_behavior'] = 'careful_following'
        
        return context
    
    def calculate_path_straightness(self, positions: List[Tuple[float, float]]) -> float:
        """Calculate how straight a path is (1.0 = perfectly straight, 0.0 = very curved)"""
        if len(positions) < 3:
            return 1.0
        
        # Calculate total path length vs straight-line distance
        total_path_length = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_path_length += math.sqrt(dx*dx + dy*dy)
        
        # Straight-line distance from start to end
        start = positions[0]
        end = positions[-1]
        straight_distance = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        
        if total_path_length == 0:
            return 1.0
        
        # Straightness is ratio of straight distance to total path length
        return straight_distance / total_path_length
    
    def log_following_metrics(self, person: PersonInfo, robot_response: dict):
        """Log following performance metrics for analysis and improvement"""
        metrics = {
            'timestamp': time.time(),
            'person_id': person.tracking_id,
            'distance': person.distance,
            'tracking_confidence': person.tracking_confidence,
            'detection_confidence': person.confidence,
            'robot_speed_m1': self.current_speed_m1,
            'robot_speed_m2': self.current_speed_m2,
            'person_velocity': math.sqrt(person.velocity_x**2 + person.velocity_y**2),
            'lateral_error': abs(person.center_x - 320),  # Distance from center
            'depth_samples': person.depth_samples if person.depth_samples else []
        }
        
        # Store metrics (could be saved to file or database)
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = []
        
        self.performance_metrics.append(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.performance_metrics) > 1000:
            self.performance_metrics = self.performance_metrics[-1000:]
    
    def handle_tracking_loss(self, person_id: int):
        """Handle when person tracking is lost"""
        if person_id in self.person_tracking_buffer:
            history = self.person_tracking_buffer[person_id]
            last_pos = history.positions[-1]
            last_vel = history.velocities[-1] if history.velocities else (0, 0)
            
            print(f"Person {person_id} tracking lost at position {last_pos}")
            
            # Predict where person might have gone
            predicted_x = last_pos[0] + last_vel[0] * 0.5  # 0.5 second prediction
            predicted_y = last_pos[1] + last_vel[1] * 0.5
            
            # Search in predicted direction first
            search_direction = math.atan2(last_vel[1], last_vel[0]) if any(last_vel) else 0
            
            print(f"Searching in direction {math.degrees(search_direction):.1f}°")
            
            # Could implement more sophisticated search patterns here
    
    def adjust_parameters_online(self, performance_metrics: dict):
        """Automatically adjust parameters based on real-time performance"""
        if not hasattr(self, 'performance_metrics') or len(self.performance_metrics) < 10:
            return
        
        recent_metrics = self.performance_metrics[-10:]
        
        # Calculate average lateral error
        avg_lateral_error = sum(m['lateral_error'] for m in recent_metrics) / len(recent_metrics)
        
        # Adjust micro-adjustment threshold based on performance
        if avg_lateral_error > 30:  # Too much error
            self.MICRO_ADJUSTMENT_THRESHOLD = max(10, self.MICRO_ADJUSTMENT_THRESHOLD - 2)
            print(f"Reducing micro-adjustment threshold to {self.MICRO_ADJUSTMENT_THRESHOLD}")
        elif avg_lateral_error < 15:  # Very accurate
            self.MICRO_ADJUSTMENT_THRESHOLD = min(50, self.MICRO_ADJUSTMENT_THRESHOLD + 1)
            print(f"Increasing micro-adjustment threshold to {self.MICRO_ADJUSTMENT_THRESHOLD}")
        
        # Adjust smoothing factor based on tracking confidence
        avg_tracking_confidence = sum(m['tracking_confidence'] for m in recent_metrics) / len(recent_metrics)
        if avg_tracking_confidence < 0.6:  # Low confidence
            self.SMOOTHING_FACTOR = min(0.5, self.SMOOTHING_FACTOR + 0.05)
            print(f"Increasing smoothing factor to {self.SMOOTHING_FACTOR:.2f}")
        elif avg_tracking_confidence > 0.9:  # High confidence
            self.SMOOTHING_FACTOR = max(0.1, self.SMOOTHING_FACTOR - 0.02)
            print(f"Decreasing smoothing factor to {self.SMOOTHING_FACTOR:.2f}")
    
    def set_forward_speed(self, qpps: int) -> None:
        """Set forward speed using project's sign convention: M1 forward is +, M2 forward is -"""
        if not self.connection_manager.ensure_connection():
            print("Cannot set speed - RoboClaw not connected")
            return
            
        roboclaw = self.connection_manager.get_roboclaw()
        if roboclaw:
            try:
                roboclaw.SpeedM1(self.address, qpps)
                roboclaw.SpeedM2(self.address, -qpps)
                self.current_speed_m1 = qpps
                self.current_speed_m2 = -qpps
            except Exception as e:
                print(f"Error setting speed: {e}")
                self.connection_manager.connected = False
    
    def set_differential_speed(self, left_speed: int, right_speed: int) -> None:
        """Set different speeds for left and right motors for smooth turning while moving"""
        if not self.connection_manager.ensure_connection():
            print("Cannot set speed - RoboClaw not connected")
            return
            
        roboclaw = self.connection_manager.get_roboclaw()
        if roboclaw:
            try:
                roboclaw.SpeedM1(self.address, left_speed)
                roboclaw.SpeedM2(self.address, right_speed)
                self.current_speed_m1 = left_speed
                self.current_speed_m2 = right_speed
            except Exception as e:
                print(f"Error setting differential speed: {e}")
                self.connection_manager.connected = False
    
    def smooth_speed_transition(self, target_left: int, target_right: int) -> None:
        """Smoothly transition to target speeds using S-curve acceleration and interpolation"""
        current_time = time.time()
        
        if self.S_CURVE_ACCEL and hasattr(self, 'last_speed_update'):
            # Use S-curve acceleration for smooth transitions
            time_elapsed = current_time - self.last_speed_update
            total_time = 0.1  # 100ms transition time
            
            # Apply S-curve acceleration to both motors
            new_left = self.s_curve_acceleration(self.current_speed_m1, target_left, time_elapsed, total_time)
            new_right = self.s_curve_acceleration(self.current_speed_m2, target_right, time_elapsed, total_time)
            
            # Set the new speeds
            self.set_differential_speed(new_left, new_right)
            self.last_speed_update = current_time
        else:
            # Fallback to original smoothing method
            left_diff = target_left - self.current_speed_m1
            right_diff = target_right - self.current_speed_m2
            
            # Limit acceleration
            left_step = max(-self.MAX_ACCELERATION, min(self.MAX_ACCELERATION, left_diff))
            right_step = max(-self.MAX_ACCELERATION, min(self.MAX_ACCELERATION, right_diff))
            
            # Apply smoothing
            new_left = self.current_speed_m1 + int(left_step * self.SMOOTHING_FACTOR)
            new_right = self.current_speed_m2 + int(right_step * self.SMOOTHING_FACTOR)
            
            # Set the new speeds
            self.set_differential_speed(new_left, new_right)
            self.last_speed_update = current_time
    
    def update_smooth_person_position(self, person_x: float, person_y: float):
        """Update smoothed person position using exponential smoothing"""
        self.smooth_person_x = (self.SMOOTHING_FACTOR * person_x + 
                               (1 - self.SMOOTHING_FACTOR) * self.smooth_person_x)
        self.smooth_person_y = (self.SMOOTHING_FACTOR * person_y + 
                               (1 - self.SMOOTHING_FACTOR) * self.smooth_person_y)
    
    def calculate_movement_command(self, person: PersonInfo, obstacles: List[ObstacleInfo]) -> Tuple[int, int]:
        """Calculate enhanced movement command using all new features"""
        current_time = time.time()
        distance = person.distance
        center_x = person.center_x
        center_y = person.center_y
        
        # Update smoothed position
        self.update_smooth_person_position(center_x, center_y)
        
        # Analyze environment context
        context = self.analyze_environment_context(person, obstacles)
        
        # Calculate adaptive following distance
        adaptive_distance = self.calculate_adaptive_distance(person)
        
        # Predict future person position with enhanced prediction
        predicted_x, predicted_y = self.predict_person_path(person, self.PREDICTION_HORIZON)
        
        # Calculate distance error using adaptive distance
        distance_error = distance - adaptive_distance
        
        # Calculate lateral error (how far person is from center)
        image_center_x = 320
        lateral_error = self.smooth_person_x - image_center_x
        
        # Generate trajectory if needed
        if not self.interpolation_active or current_time - self.last_trajectory_update > 0.5:
            current_pos = (self.smooth_person_x, self.smooth_person_y)
            target_pos = (predicted_x, predicted_y)
            current_vel = (0, 0)  # Current robot velocity
            target_vel = (0, 0)   # Target velocity
            
            self.current_trajectory = self.generate_trajectory(current_pos, target_pos, current_vel, target_vel)
            self.trajectory_start_time = current_time
            self.last_trajectory_update = current_time
            self.interpolation_active = True
        
        # Interpolate along trajectory
        if self.interpolation_active:
            interp_x, interp_y = self.interpolate_trajectory(current_time)
            # Use interpolated position for more accurate following
            lateral_error = interp_x - image_center_x
        
        # Base forward speed based on adaptive distance and velocity matching
        if distance < adaptive_distance - 0.2:
            # Way too close - back up slowly for safety
            base_forward = -self.FORWARD_TARGET_QPPS // 4
            print(f"  Too close ({distance:.2f}m < {adaptive_distance:.2f}m) - backing up slowly")
        elif distance < adaptive_distance:
            # Slightly close - move forward slowly
            base_forward = self.FORWARD_TARGET_QPPS // 4
            print(f"  Close ({distance:.2f}m < {adaptive_distance:.2f}m) - moving forward slowly")
        elif distance > adaptive_distance + 0.3:
            # Too far - move forward quickly to catch up
            target_speed = self.match_person_velocity(person)
            base_forward = target_speed
            print(f"  Too far ({distance:.2f}m > {adaptive_distance:.2f}m) - moving forward quickly to catch up")
        else:
            # Good distance - maintain matched speed
            base_forward = self.match_person_velocity(person)
            print(f"  Good distance ({distance:.2f}m ≈ {adaptive_distance:.2f}m) - maintaining speed")
        
        # Apply S-curve acceleration if enabled
        if self.S_CURVE_ACCEL:
            time_elapsed = current_time - self.trajectory_start_time
            base_forward = self.s_curve_acceleration(self.current_speed_m1, base_forward, 
                                                   time_elapsed, self.TRAJECTORY_TIME)
        
        # CRITICAL FIX: Ensure base_forward is never negative for forward movement
        if base_forward < 0:
            print(f"  WARNING: base_forward was negative ({base_forward}), forcing to positive")
            base_forward = abs(base_forward)
        
        # Calculate turning adjustment with enhanced smoothing
        if abs(lateral_error) > self.MICRO_ADJUSTMENT_THRESHOLD:
            # Need to turn - calculate turn speed
            turn_intensity = min(abs(lateral_error) / 100.0, 1.0)  # Normalize to 0-1
            
            # Apply cubic spline smoothing to turn intensity
            if self.CUBIC_SPLINE_SMOOTHING:
                turn_intensity = self.cubic_spline_interpolation([0, 50, 100], [0, 0.5, 1.0], abs(lateral_error))
            
            # Adjust turn speed based on context
            if context['person_making_turn']:
                turn_speed = int(self.MICRO_TURN_QPPS * turn_intensity * 1.5)  # More responsive for turns
            else:
                turn_speed = int(self.MICRO_TURN_QPPS * turn_intensity)
            
            if lateral_error > 0:
                # Person is to the right - turn right (left wheel faster)
                left_speed = base_forward + turn_speed
                right_speed = -(base_forward - turn_speed)  # M2 is inverted
                print(f"  Person to right ({lateral_error:.1f}px) - turning right while moving forward")
            else:
                # Person is to the left - turn left (right wheel faster)
                left_speed = base_forward - turn_speed
                right_speed = -(base_forward + turn_speed)  # M2 is inverted
                print(f"  Person to left ({abs(lateral_error):.1f}px) - turning left while moving forward")
        else:
            # No turning needed - straight movement
            left_speed = base_forward
            right_speed = -base_forward  # M2 is inverted
            print(f"  Person centered - moving straight forward")
        
        # Apply context-based adjustments
        if context['recommended_behavior'] == 'cautious_following':
            # Reduce speed in high-obstacle environments
            left_speed = int(left_speed * 0.7)
            right_speed = int(right_speed * 0.7)
            print(f"  High obstacle density - reducing speed to 70%")
        
        # FINAL SAFETY CHECK: Ensure we're not going backwards when we should be going forward
        if distance > adaptive_distance and left_speed < 0:
            print(f"  CRITICAL ERROR: Person is in front but robot trying to go backwards!")
            print(f"  Distance: {distance:.2f}m, Target: {adaptive_distance:.2f}m")
            print(f"  Forcing forward movement...")
            left_speed = abs(left_speed)  # Force positive
            right_speed = -abs(right_speed)  # Keep M2 inverted but positive magnitude
        
        # Print final movement summary
        movement_direction = "FORWARD" if left_speed > 0 else "BACKWARD" if left_speed < 0 else "STOPPED"
        print(f"  Final command: {movement_direction} | Left: {left_speed}, Right: {right_speed}")
        
        return left_speed, right_speed
    
    def s_curve_acceleration(self, current_speed: int, target_speed: int, time_elapsed: float, total_time: float) -> int:
        """Generate S-curve acceleration profile for smooth speed transitions"""
        if total_time <= 0:
            return target_speed
            
        # Normalize time to 0-1
        t = min(time_elapsed / total_time, 1.0)
        
        # S-curve function: smooth start, linear middle, smooth end
        if t < 0.5:
            # First half: smooth acceleration
            s = 2 * t * t
        else:
            # Second half: smooth deceleration
            s = 1 - 2 * (1 - t) * (1 - t)
        
        # Interpolate between current and target speeds
        interpolated_speed = current_speed + int((target_speed - current_speed) * s)
        return interpolated_speed
    
    def cubic_spline_interpolation(self, x_points: list, y_points: list, x_new: float) -> float:
        """Interpolate using cubic splines for smooth path following"""
        if len(x_points) < 4:
            # Fallback to linear interpolation if not enough points
            return np.interp(x_new, x_points, y_points)
        
        try:
            # Create cubic spline
            cs = CubicSpline(x_points, y_points, bc_type='natural')
            return float(cs(x_new))
        except:
            # Fallback to linear interpolation
            return np.interp(x_new, x_points, y_points)
    
    def predict_person_position(self, person: PersonInfo, time_horizon: float) -> Tuple[float, float]:
        """Predict future person position using velocity and acceleration analysis"""
        current_time = time.time()
        
        # Add current position to history
        self.position_history.append((current_time, person.center_x, person.center_y))
        
        if len(self.position_history) < 3:
            return person.center_x, person.center_y
        
        # Calculate velocity from recent positions
        recent_positions = list(self.position_history)[-5:]
        if len(recent_positions) >= 2:
            dt = recent_positions[-1][0] - recent_positions[-2][0]
            if dt > 0:
                vx = (recent_positions[-1][1] - recent_positions[-2][1]) / dt
                vy = (recent_positions[-1][2] - recent_positions[-2][2]) / dt
                
                # Predict future position
                predicted_x = person.center_x + vx * time_horizon
                predicted_y = person.center_y + vy * time_horizon
                
                return predicted_x, predicted_y
        
        return person.center_x, person.center_y
    
    def generate_trajectory(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float], 
                           current_vel: Tuple[float, float], target_vel: Tuple[float, float]) -> list:
        """Generate smooth trajectory using cubic splines and S-curve acceleration"""
        if not self.CUBIC_SPLINE_SMOOTHING:
            # Simple linear trajectory
            return self.generate_linear_trajectory(current_pos, target_pos)
        
        # Create time points for trajectory
        time_points = np.linspace(0, self.TRAJECTORY_TIME, self.TRAJECTORY_POINTS)
        
        # Generate smooth path using cubic splines
        x_points = [current_pos[0], target_pos[0]]
        y_points = [current_pos[1], target_pos[1]]
        
        # Add intermediate control points for smoother curves
        mid_x = (current_pos[0] + target_pos[0]) / 2
        mid_y = (current_pos[1] + target_pos[1]) / 2
        
        # Add slight curve for natural movement
        curve_strength = 0.1
        curve_x = mid_x + (target_pos[1] - current_pos[1]) * curve_strength
        curve_y = mid_y - (target_pos[0] - current_pos[0]) * curve_strength
        
        x_points = [current_pos[0], curve_x, target_pos[0]]
        y_points = [current_pos[1], curve_y, target_pos[1]]
        
        # Create cubic spline for path
        try:
            path_spline = CubicSpline([0, 0.5, 1], [x_points, y_points], axis=1)
            
            # Generate trajectory points
            trajectory = []
            for t in time_points:
                t_norm = t / self.TRAJECTORY_TIME
                pos = path_spline(t_norm)
                trajectory.append((float(pos[0]), float(pos[1])))
            
            return trajectory
            
        except:
            # Fallback to linear trajectory
            return self.generate_linear_trajectory(current_pos, target_pos)
    
    def generate_linear_trajectory(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> list:
        """Generate simple linear trajectory as fallback"""
        trajectory = []
        for i in range(self.TRAJECTORY_POINTS):
            t = i / (self.TRAJECTORY_POINTS - 1)
            x = current_pos[0] + t * (target_pos[0] - current_pos[0])
            y = current_pos[1] + t * (target_pos[1] - current_pos[1])
            trajectory.append((x, y))
        return trajectory
    
    def interpolate_trajectory(self, current_time: float) -> Tuple[float, float]:
        """Interpolate current position along the planned trajectory"""
        if not self.current_trajectory or len(self.current_trajectory) < 2:
            return 0, 0
        
        time_elapsed = current_time - self.trajectory_start_time
        trajectory_time = self.TRAJECTORY_TIME
        
        if time_elapsed >= trajectory_time:
            # Trajectory complete
            final_pos = self.current_trajectory[-1]
            self.interpolation_active = False
            return final_pos
        
        # Normalize time to 0-1
        t = time_elapsed / trajectory_time
        
        # Use cubic spline interpolation for smooth movement
        if len(self.current_trajectory) >= 4:
            # Create time points for spline
            time_points = np.linspace(0, 1, len(self.current_trajectory))
            x_points = [pos[0] for pos in self.current_trajectory]
            y_points = [pos[1] for pos in self.current_trajectory]
            
            try:
                # Interpolate X and Y separately
                x_spline = CubicSpline(time_points, x_points)
                y_spline = CubicSpline(time_points, y_points)
                
                x_interp = float(x_spline(t))
                y_interp = float(y_spline(t))
                
                return x_interp, y_interp
            except:
                # Fallback to linear interpolation
                pass
        
        # Linear interpolation fallback
        idx = int(t * (len(self.current_trajectory) - 1))
        if idx >= len(self.current_trajectory) - 1:
            return self.current_trajectory[-1]
        
        # Linear interpolation between two points
        t_local = t * (len(self.current_trajectory) - 1) - idx
        pos1 = self.current_trajectory[idx]
        pos2 = self.current_trajectory[idx + 1]
        
        x_interp = pos1[0] + t_local * (pos2[0] - pos1[0])
        y_interp = pos1[1] + t_local * (pos2[1] - pos1[1])
        
        return x_interp, y_interp
    
    def stop_both(self) -> None:
        """Stop both motors"""
        if not self.connection_manager.ensure_connection():
            print("Cannot stop motors - RoboClaw not connected")
            return
            
        roboclaw = self.connection_manager.get_roboclaw()
        if roboclaw:
            try:
                roboclaw.SpeedM1(self.address, 0)
                roboclaw.SpeedM2(self.address, 0)
                self.current_speed_m1 = 0
                self.current_speed_m2 = 0
            except Exception as e:
                print(f"Error stopping motors: {e}")
                self.connection_manager.connected = False
    
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
        if not self.connection_manager.ensure_connection():
            print("Cannot turn - RoboClaw not connected")
            return
            
        roboclaw = self.connection_manager.get_roboclaw()
        if not roboclaw:
            return
            
        try:
            # Differential turn-in-place left using the project's sign convention
            # Left wheel backward (M1 negative), right wheel forward (M2 negative)
            roboclaw.SpeedM1(self.address, -turn_qpps)
            roboclaw.SpeedM2(self.address, -turn_qpps)
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
        except Exception as e:
            print(f"Error during turn: {e}")
            self.connection_manager.connected = False
            self.stop_both()
    
    def follow_person(self, person: PersonInfo, obstacles: List[ObstacleInfo]) -> None:
        """Follow a detected person maintaining safe distance with enhanced movement"""
        distance = person.distance
        center_x = person.center_x
        adaptive_distance = self.calculate_adaptive_distance(person)
        
        print(f"FOLLOWING PERSON:")
        print(f"  Position: {center_x:.1f}px from center")
        print(f"  Distance: {distance:.2f}m (target: {adaptive_distance:.2f}m)")
        print(f"  Velocity: {math.sqrt(person.velocity_x**2 + person.velocity_y**2):.1f} px/s")
        print(f"  Tracking ID: {person.tracking_id} (confidence: {person.tracking_confidence:.2f})")
        
        # Calculate enhanced movement command with obstacles context
        left_speed, right_speed = self.calculate_movement_command(person, obstacles)
        
        # Apply smooth speed transition
        self.smooth_speed_transition(left_speed, right_speed)
        
        # Update last known position
        self.last_person_center_x = center_x
        self.last_person_center_y = person.center_y
    
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
                # Check RoboClaw connection status
                if not self.connection_manager.is_connected():
                    print("RoboClaw disconnected, waiting for reconnection...")
                    self.stop_both()  # Ensure motors are stopped
                    if not self.connection_manager.wait_for_connection():
                        print("Failed to reconnect, retrying...")
                        time.sleep(2.0)
                        continue
                    print("RoboClaw reconnected successfully!")
                
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
                    
                    if closest_person.confidence > self.confidence_threshold:  # Use enhanced confidence threshold
                        self.person_detected = True
                        self.last_person_center_x = closest_person.center_x
                        self.follow_person(closest_person, obstacles)  # Pass obstacles for context
                    else:
                        self.person_detected = False
                        self.smooth_speed_transition(0, 0)  # Smooth stop
                else:
                    # No people detected
                    if self.person_detected:
                        print("Person lost, stopping smoothly")
                        self.person_detected = False
                        self.smooth_speed_transition(0, 0)  # Smooth stop
                    else:
                        # No person, no obstacles - slowly spin to find someone
                        print("No person detected, slowly spinning to search...")
                        roboclaw = self.connection_manager.get_roboclaw()
                        if roboclaw:
                            roboclaw.SpeedM1(self.address, -self.TURN_QPPS // 4)  # Slow left turn
                            roboclaw.SpeedM2(self.address, -self.TURN_QPPS // 4)
                            time.sleep(3)  # Spin for 3 seconds
                            self.smooth_speed_transition(0, 0)  # Smooth stop
                
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
        connection_status = "Connected" if self.connection_manager.is_connected() else "Disconnected"
        status_text = f"RoboClaw: {connection_status} | People: {len(people)}, Obstacles: {len(obstacles)}"
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
            self.connection_manager.close()
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
