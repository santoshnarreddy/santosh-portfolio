"""
Attendance logging utilities.
Author: Santosh Narreddy

Handles CSV writing, session management, and report printing.
"""

import os
import csv
from datetime import datetime


class AttendanceLogger:

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir    = log_dir
        self.session_dt = None
        self.log_path   = None
        self.records    = []   # [{name, time, date}]
        os.makedirs(log_dir, exist_ok=True)

    def start_session(self):
        self.session_dt = datetime.now()
        date_str        = self.session_dt.strftime('%Y-%m-%d')
        self.log_path   = os.path.join(self.log_dir, f"attendance_{date_str}.csv")

        # Create file with header if it doesn't exist yet
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Name', 'Date', 'Time', 'Status'])
                writer.writeheader()

        print(f"[INFO] Attendance log: {self.log_path}")

    def mark_attendance(self, name: str):
        """Append a single attendance record to the CSV."""
        now   = datetime.now()
        record = {
            'Name':   name,
            'Date':   now.strftime('%Y-%m-%d'),
            'Time':   now.strftime('%H:%M:%S'),
            'Status': 'Present'
        }
        self.records.append(record)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Date', 'Time', 'Status'])
            writer.writerow(record)

    def save_session(self):
        """Nothing extra to do — records are written in real-time."""
        print(f"[INFO] {len(self.records)} attendance records saved to {self.log_path}")

    def print_report(self):
        """Print a summary of today's attendance to the console."""
        if not self.records:
            print("[INFO] No attendance recorded this session.")
            return

        print(f"\n{'─'*45}")
        print(f"  Attendance Report — {self.session_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'─'*45}")
        print(f"  {'Name':<20} {'Time':>10}")
        print(f"{'─'*45}")
        for r in self.records:
            print(f"  {'✓ ' + r['Name']:<20} {r['Time']:>10}")
        print(f"{'─'*45}")
        print(f"  Total present: {len(self.records)}")
        print(f"{'─'*45}\n")
