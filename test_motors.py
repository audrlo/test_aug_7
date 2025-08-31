from roboclaw import Roboclaw
import time

roboclaw = Roboclaw("/dev/ttyACM0", 38400)
roboclaw.Open()

address = 0x80

result, version = roboclaw.ReadVersion(address)
print("Success:", result)
print("Version:", version)

# We've determined ForwardM1 is not working, so we're not using it.
# roboclaw.ForwardM1(address, 64)  # Half speed
# print("Move???")
# time.sleep(2)
# roboclaw.ForwardM1(address, 0)   # Stop
# print("stopped")

# Check voltage
status, volts = roboclaw.ReadMainBatteryVoltage(address)
print("Battery voltage:", volts / 10.0, "V")

# Try using signed duty mode
print("Trying DutyM1...")
for i in range(20):
    roboclaw.DutyM1(address, 819 * i)

    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)
time.sleep(5)
for i in range(20):
    roboclaw.DutyM1(address, 819 * (20 -i))

    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)
roboclaw.DutyM1(address, 0)

print("Done.")

# Try using signed duty mode
print("Trying DutyM2...")
for i in range(20):

    roboclaw.DutyM2(address, -819 * i)
    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)
time.sleep(5)
for i in range(20):

    roboclaw.DutyM2(address, -819 * (20 -i))
    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)

roboclaw.DutyM2(address, 0)
print("Done.")

time.sleep(5)

print("Trying Speed")
for i in range(20):
    roboclaw.SpeedM1(address, 250 * i)
    roboclaw.SpeedM2(address, -250 * i)
    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)
time.sleep(5)
for i in range(20):
    roboclaw.SpeedM1(address, 250 * (20 -i))
    roboclaw.SpeedM2(address, -250 * (20 -i))
    print(roboclaw.ReadSpeedM1(address)[1] + roboclaw.ReadSpeedM2(address)[1])
    time.sleep(0.05)
roboclaw.SpeedM1(address, 0)
roboclaw.SpeedM2(address, 0)
print("Done")
