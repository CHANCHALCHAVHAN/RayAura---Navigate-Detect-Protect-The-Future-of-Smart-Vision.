import datetime
import csv

# 📌 Function to log detected objects
def log_detection(object_name, confidence, timestamp):
    log_entry = [object_name, confidence, timestamp]
    
    with open("detection_logs.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)

# 📌 Example Usage
if __name__ == "__main__":
    print("📊 Logging Started...")
    log_detection("Car", 0.89, datetime.datetime.now())
    log_detection("Person", 0.75, datetime.datetime.now())
    print("✅ Logs Updated in detection_logs.csv")
