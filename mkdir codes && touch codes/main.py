mkdir codes && touch codes/main.py

# main.py - Rayaura AI-Powered Monitoring System

# 📌 Import necessary libraries
import cv2  # OpenCV for object detection
import numpy as np  # Numerical operations
import time  # Time handling
from ultralytics import YOLO  # YOLO model for real-time object detection

# 📌 Load pre-trained YOLO model (you can replace with your trained model)
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano for efficiency

# 📌 Open webcam or video file for real-time monitoring
video_source = 0  # Use 0 for webcam, or replace with a video file path
cap = cv2.VideoCapture(video_source)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("🚀 Starting Rayaura AI Monitoring System...")

# 📌 Object detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended or error in reading frame.")
        break

    # 📌 Run YOLO object detection
    results = model(frame)

    # 📌 Draw detected objects on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Object class name

            # Draw rectangle & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 📌 Show the output frame
    cv2.imshow("Rayaura - Real-time Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 📌 Release resources
cap.release()
cv2.destroyAllWindows()
print("🔴 Rayaura Monitoring System Stopped.")


