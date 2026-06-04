from services.camera_service import VitalsCameraService

def main():
    service = VitalsCameraService()
    try:
        service.run()
    except KeyboardInterrupt:
        print("\n[!] Engine shutting down safely.")

if __name__ == "__main__":
    main()