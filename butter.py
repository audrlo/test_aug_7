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
        self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
        
        # Control parameters (following avoid_obstacle.py conventions)
        self.RAMP_STEP_QPPS = 20  # Smaller steps for smoother ramping
        self.FORWARD_TARGET_QPPS = 50 * 20  # ~"half speed" target
        self.TURN_QPPS = 50 * 10  # turning speed magnitude
        self.RAMP_STEP_DELAY_S = 0.02  # Faster updates for smoother movement
        self.OBSTACLE_METERS = 1.0  # minimum safe distance
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
    
    def calculate_movement_command(self, person: PersonInfo) -> Tuple[int, int]:
        """Calculate smooth movement command using full interpolation system"""
        current_time = time.time()
        distance = person.distance
        center_x = person.center_x
        center_y = person.center_y
        
        # Update smoothed position
        self.update_smooth_person_position(center_x, center_y)
        
        # Predict future person position
        predicted_x, predicted_y = self.predict_person_position(person, self.PREDICTION_HORIZON)
        
        # Calculate distance error
        distance_error = distance - self.PERSON_FOLLOW_DISTANCE
        
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
        
        # Base forward speed based on distance with S-curve acceleration
        if distance < self.PERSON_FOLLOW_DISTANCE:
            # Too close - back up slowly
            base_forward = -self.FORWARD_TARGET_QPPS // 3
        elif distance > self.PERSON_FOLLOW_DISTANCE + 0.3:
            # Too far - move forward
            base_forward = self.FORWARD_TARGET_QPPS
        else:
            # Good distance - maintain speed
            base_forward = self.FORWARD_TARGET_QPPS // 2
        
        # Apply S-curve acceleration if enabled
        if self.S_CURVE_ACCEL:
            time_elapsed = current_time - self.trajectory_start_time
            base_forward = self.s_curve_acceleration(self.current_speed_m1, base_forward, 
                                                   time_elapsed, self.TRAJECTORY_TIME)
        
        # Calculate turning adjustment with cubic spline smoothing
        if abs(lateral_error) > self.MICRO_ADJUSTMENT_THRESHOLD:
            # Need to turn - calculate turn speed
            turn_intensity = min(abs(lateral_error) / 100.0, 1.0)  # Normalize to 0-1
            
            # Apply cubic spline smoothing to turn intensity
            if self.CUBIC_SPLINE_SMOOTHING:
                turn_intensity = self.cubic_spline_interpolation([0, 50, 100], [0, 0.5, 1.0], abs(lateral_error))
            
            turn_speed = int(self.MICRO_TURN_QPPS * turn_intensity)
            
            if lateral_error > 0:
                # Person is to the right - turn right (left wheel faster)
                left_speed = base_forward + turn_speed
                right_speed = -(base_forward - turn_speed)  # M2 is inverted
            else:
                # Person is to the left - turn left (right wheel faster)
                left_speed = base_forward - turn_speed
                right_speed = -(base_forward + turn_speed)  # M2 is inverted
        else:
            # No turning needed - straight movement
            left_speed = base_forward
            right_speed = -base_forward  # M2 is inverted
        
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
    
    def follow_person(self, person: PersonInfo) -> None:
        """Follow a detected person maintaining safe distance with smooth movement"""
        distance = person.distance
        center_x = person.center_x
        
        print(f"Following person at {distance:.2f}m, center_x: {center_x:.1f}")
        
        # Calculate smooth movement command
        left_speed, right_speed = self.calculate_movement_command(person)
        
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
                    
                    if closest_person.confidence > 0.5:  # confidence threshold
                        self.person_detected = True
                        self.last_person_center_x = closest_person.center_x
                        self.follow_person(closest_person)
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
