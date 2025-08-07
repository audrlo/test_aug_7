#!/usr/bin/env python3
from roboclaw import Roboclaw
import pyrealsense2 as rs
import numpy as np
import cv2
import time

roboclaw = Roboclaw("/dev/ttyACM0", 115200)
roboclaw.Open()

address = 0x80

result, version = roboclaw.ReadVersion(address)
print("Success:", result)
print("Version:", version)

def get_center_distance(depth_frame: rs.depth_frame) -> float:
    """Return distance at image center in meters. Returns float('inf') if invalid."""
    if not depth_frame:
        return float("inf")
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    distance_meters = depth_frame.get_distance(width // 2, height // 2)
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
    roboclaw.SpeedM1(address, -turn_qpps)
    roboclaw.SpeedM2(address, -turn_qpps)
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
    roboclaw = Roboclaw("/dev/ttyACM0", 115200)
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
    RAMP_STEP_QPPS = 500  # same increment used in test_version_2.py loops
    FORWARD_TARGET_QPPS = 500 * 20  # treat as ~"half speed" target in QPPS
    TURN_QPPS = 500 * 10  # turning speed magnitude
    RAMP_STEP_DELAY_S = 0.05
    OBSTACLE_METERS = 1.0
    CHECK_INTERVAL_S = 0.05

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
                # Keep commanding forward at target (ensures recovery if any slip)
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

