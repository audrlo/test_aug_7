# Raspberry Pi Packet Serial - Python 3

This repository contains the Python 3 compatible version of the test file 'packet_serial.py'
and the BasicMicro Python library 'roboclaw.py'. The test file
operates a RoboClaw in packet serial mode with a Raspberry Pi single board computer.

## Python 3 Compatibility

This code has been updated from the original Python 2 version to be fully compatible with Python 3. Key changes include:

- Replaced `chr()` and `ord()` functions with proper bytes handling for serial communication
- Converted `long()` to `int()` for numeric operations
- Fixed string concatenation with bytes in version reading
- Corrected method calls and parameter passing issues

## Original Documentation

The accompanying Application Note can be [found here](https://resources.basicmicro.com/packet-serial-with-the-raspberry-pi-3/).

## Quick start (Python 3)

1. Create and activate a virtual environment (recommended):

   - macOS/Linux:
     ```sh
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     py -3 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Run any script with Python 3:

   ```sh
   python3 test_combo.py
   ```
