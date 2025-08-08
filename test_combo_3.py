from roboclaw import Roboclaw
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import random

# does this need to be here?

roboclaw = Roboclaw("/dev/ttyACM0", 38400) # should this be /dev/ttyUSB0? and 38400?
roboclaw.Open()

address = 0x80

result, version = roboclaw.ReadVersion(address) # version check to see if it's alive
print("Success:", result) # should be 1
print("Version:", version)
print("Starting test_combo_2.py: basic obstacle avoidance")

def get_center_distance(depth_frame: rs.depth_frame) -> float:
    """Return distance at image center in meters. Returns float('inf') if invalid."""
    if not depth_frame:
        return float("inf")
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    distance_meters = 6.0
    for i in range(21):
        for j in range(11):
            x = (width // 2) - 10 + i
            y = (height // 2) - 5 + j
            distance_meters = min(depth_frame.get_distance(x, y), distance_meters)

    return distance_meters if distance_meters > 0 else float("inf")


def set_forward_speed(roboclaw: Roboclaw, address: int, qpps: int) -> None:
    # Match test_version_2.py convention: M1 forward is +, M2 forward is -
    roboclaw.SpeedM1(address, qpps)
    roboclaw.SpeedM2(address, -qpps)


def stop_both(roboclaw: Roboclaw, address: int) -> None:
    roboclaw.SpeedM1(address, 0)
    roboclaw.SpeedM2(address, 0)


def ramp_forward_to_half_speed(roboclaw: Roboclaw, address: int, target_qpps: int, step_qpps: int, step_delay_s: float) -> None:
    for qpps in range(0, target_qpps + 1, step_qpps):
        set_forward_speed(roboclaw, address, qpps)
        time.sleep(step_delay_s)

#Rename later to 'turn randomly'

def turn_left_until_clear(
    roboclaw: Roboclaw,
    address: int,
    pipeline: rs.pipeline,
    turn_qpps: int,
    obstacle_meters: float,
    poll_interval_s: float,
) -> None:
    # Differential turn-in-place left using the project's sign convention
    # Left wheel backward (M1 negative), right wheel forward (M2 negative)
    # Half speed (of half speed bruh) for slow turning. Experiment with hard coded values in line 58-59 and line 61.

    #Random number to choose direction
    direction = random.randint(0, 1)*2 - 1
    
    roboclaw.SpeedM1(address, direction * turn_qpps) 
    roboclaw.SpeedM2(address, direction * turn_qpps) 
    print("Turning left until clear")
    time.sleep(1) # wait 1 second to let the robot turn left
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        distance = get_center_distance(depth)
        if distance > obstacle_meters:
            break
        time.sleep(poll_interval_s)
    stop_both(roboclaw, address)


def main() -> None:
    # Motor controller setup
    roboclaw = Roboclaw("/dev/ttyACM0", 38400)
    roboclaw.Open()
    address = 0x80

    result, version = roboclaw.ReadVersion(address)
    print("Roboclaw comms:", result, "Version:", version)

    # RealSense depth pipeline setup (based on test_cam.py)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # Control parameters (mirroring increments from test_version_2.py)
    RAMP_STEP_QPPS = 50  # same increment used in test_version_2.py loops, this was 500 before
    FORWARD_TARGET_QPPS = RAMP_STEP_QPPS * 20  # treat as ~"half speed" target in QPPS, this was 500 before
    TURN_QPPS = RAMP_STEP_QPPS * 10  # turning speed magnitude, this was 500 before
    RAMP_STEP_DELAY_S = 0.05
    OBSTACLE_METERS = 1.0
    SLOW_METERS = 3.0
    CHECK_INTERVAL_S = 0.05
    SLOW_FORWARD_TARGET_QPPS = 500

    try:
        # Initial forward ramp
        ramp_forward_to_half_speed(roboclaw, address, FORWARD_TARGET_QPPS, RAMP_STEP_QPPS, RAMP_STEP_DELAY_S)

        while True:
            # Read center depth and react
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            center_distance = get_center_distance(depth)
            print(f"Center depth: {center_distance:.2f} m")

            if center_distance <= OBSTACLE_METERS:
                # Obstacle ahead: turn left until clear, then ramp forward again
                stop_both(roboclaw, address)
                turn_left_until_clear(
                    roboclaw,
                    address,
                    pipeline,
                    TURN_QPPS,
                    OBSTACLE_METERS,
                    CHECK_INTERVAL_S,
                )
                ramp_forward_to_half_speed(roboclaw, address, FORWARD_TARGET_QPPS, RAMP_STEP_QPPS, RAMP_STEP_DELAY_S)
            else:
                if center_distance <= SLOW_METERS:
                    set_forward_speed(roboclaw, address, SLOW_FORWARD_TARGET_QPPS)
                else:
                    set_forward_speed(roboclaw, address, FORWARD_TARGET_QPPS)
                

            time.sleep(CHECK_INTERVAL_S)

    except KeyboardInterrupt:
        print("Stopping due to KeyboardInterrupt...")
    finally:
        try:
            stop_both(roboclaw, address)
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()

