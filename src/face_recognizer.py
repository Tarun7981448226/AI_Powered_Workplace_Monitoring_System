"""Standalone live recognizer: shows names and confidence on the camera feed.

Uses the same recognizer, detector and tracking as the main monitoring system,
so what you see here is exactly what an attendance session will use.
Press ESC in the camera window to stop.
"""
from monitor import run_monitoring


if __name__ == "__main__":
    run_monitoring(total_seconds=24 * 3600, show=True)
