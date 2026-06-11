import csv
import os
from datetime import datetime
from config.settings import CSV_LOG_FILE

class VitalsLogger:
    def __init__(self):
        self.filepath = CSV_LOG_FILE
        
        # Ensure the 'data' directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # If the file is brand new, write the headers first
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Heart_Rate_BPM", "HRV_RMSSD_ms", "Breathing_Rate_BPM"])

    def log_metrics(self, metrics_dict):
        """Appends a new row of metrics with the current system time."""
        # We only want to log if we actually have at least a Heart Rate
        if not metrics_dict.get('hr') or metrics_dict['hr'] == '--':
            return
            
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            writer.writerow([
                timestamp,
                metrics_dict.get('hr', ''),
                metrics_dict.get('hrv', ''),
                metrics_dict.get('br', '')
            ])