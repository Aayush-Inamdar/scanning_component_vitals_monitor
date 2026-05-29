"""Entry point for the camera-based vitals monitoring application."""

import sys

import cv2

from services.camera_service import VitalsMonitor


def main() -> None:
    """Initialize the app, run monitoring, and release resources on exit."""
    sys.stdout.reconfigure(encoding="utf-8")

    monitor = VitalsMonitor()
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
