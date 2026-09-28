"""
PyInstaller Desktop Application Builder for Hand Gesture Controller
Author: Santosh Narreddy
Packages gesture_controller.py into a standalone desktop executable (.app / .exe).
"""
import os
import sys
import subprocess

def build():
    print("[INFO] Packaging Hand Gesture Controller with PyInstaller...")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=HandGestureController",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--exclude-module=tensorflow",
        "--exclude-module=torch",
        "gesture_controller.py"
    ]
    subprocess.run(cmd, check=True)
    print("[SUCCESS] Standalone executable created in: dist/HandGestureController")

if __name__ == '__main__':
    build()
