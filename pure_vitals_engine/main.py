from services.camera_service import VitalsCameraService

def main():
    service = VitalsCameraService()
    try:
        service.run()
    except KeyboardInterrupt:
        print("\n[!] Engine interrupted by user. Shutting down safely.")
        service.generate_and_save_graph()

if __name__ == "__main__":
    main()