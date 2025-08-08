from roboclaw import Roboclaw
from time import sleep

def read_encoder_values(roboclaw, address):
    """Read and display encoder values for both motors"""
    # Read Motor 1 encoder
    enc1 = roboclaw.ReadEncM1(address)
    if enc1[0]:  # Check if read was successful
        print(f"  Motor 1 Encoder: {enc1[1]} counts, Status: {enc1[2]}")
    else:
        print("  Motor 1 Encoder: Read failed")
    
    # Read Motor 2 encoder
    enc2 = roboclaw.ReadEncM2(address)
    if enc2[0]:  # Check if read was successful
        print(f"  Motor 2 Encoder: {enc2[1]} counts, Status: {enc2[2]}")
    else:
        print("  Motor 2 Encoder: Read failed")

def read_speed_values(roboclaw, address):
    """Read and display speed values for both motors"""
    # Read Motor 1 speed
    speed1 = roboclaw.ReadSpeedM1(address)
    if speed1[0]:  # Check if read was successful
        print(f"  Motor 1 Speed: {speed1[1]} QPPS (Quadrature Pulses Per Second)")
    else:
        print("  Motor 1 Speed: Read failed")
    
    # Read Motor 2 speed
    speed2 = roboclaw.ReadSpeedM2(address)
    if speed2[0]:  # Check if read was successful
        print(f"  Motor 2 Speed: {speed2[1]} QPPS (Quadrature Pulses Per Second)")
    else:
        print("  Motor 2 Speed: Read failed")

if __name__ == "__main__":
    
    address = 0x80
    roboclaw = Roboclaw("/dev/ttyS0", 38400)
    
    print("Initializing RoboClaw connection...")
    if roboclaw.Open():
        print("✓ RoboClaw connection opened successfully")
    else:
        print("✗ Failed to open RoboClaw connection")
        exit(1)
    
    # Read version to confirm communication
    version = roboclaw.ReadVersion(address)
    if version[0]:
        print(f"✓ RoboClaw Version: {version[1]}")
    else:
        print("✗ Failed to read RoboClaw version")
    
    print("\nStarting motor control loop...\n")
    
    loop_count = 0
    while True:
        loop_count += 1
        print(f"=== Loop {loop_count} ===")
        
        # Motor 1 Forward
        print("Sending command to RoboClaw: Motor 1 Forward (speed 64)")
        roboclaw.ForwardM1(address, 64)
        sleep(0.5)  # Short delay to let motor start
        read_encoder_values(roboclaw, address)
        read_speed_values(roboclaw, address)
        sleep(1.5)  # Continue running
        
        # Motor 1 Stop
        print("Sending command to RoboClaw: Motor 1 Stop")
        roboclaw.ForwardM1(address, 0)
        sleep(0.5)  # Short delay
        read_encoder_values(roboclaw, address)
        read_speed_values(roboclaw, address)
        sleep(1.5)
        
        # Motor 2 Forward
        print("Sending command to RoboClaw: Motor 2 Forward (speed 64)")
        roboclaw.ForwardM2(address, 64)
        sleep(0.5)  # Short delay to let motor start
        read_encoder_values(roboclaw, address)
        read_speed_values(roboclaw, address)
        sleep(1.5)  # Continue running
        
        # Motor 2 Stop
        print("Sending command to RoboClaw: Motor 2 Stop")
        roboclaw.ForwardM2(address, 0)
        sleep(0.5)  # Short delay
        read_encoder_values(roboclaw, address)
        read_speed_values(roboclaw, address)
        sleep(1.5)
        
        print()  # Empty line for readability
    
    

