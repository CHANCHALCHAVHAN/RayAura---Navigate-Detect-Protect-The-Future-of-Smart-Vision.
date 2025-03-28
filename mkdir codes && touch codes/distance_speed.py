import cv2
import numpy as np
import time

# Constants
FOCAL_LENGTH = 700  # Adjust based on camera calibration
REAL_WIDTH = 2  # Approximate width of a car (in meters)

# Load pre-trained YOLO model
model = cv2.dnn.readNet("yolov8n.weights", "yolov8n.cfg")

# Open video stream
cap = cv2.VideoCapture("drive_monitoring_video.mp4")

prev_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]
    detections = model.forward(output_layers)

    current_positions = {}
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Object detected
                center_x, center_y, w, h = (obj[:4] * [width, height, width, height]).astype(int)
                distance = (FOCAL_LENGTH * REAL_WIDTH) / w  # Estimate distance

                if class_id == 2:  # If the detected object is a car
                    current_positions[class_id] = (center_x, distance, time.time())

                    if class_id in prev_positions:
                        x1, d1, t1 = prev_positions[class_id]
                        x2, d2, t2 = current_positions[class_id]

                        speed = (abs(d2 - d1) / (t2 - t1)) * 3.6  # Convert to km/h
                        cv2.putText(frame, f"Speed: {speed:.2f} km/h", (center_x, center_y - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                prev_positions = current_positions

    cv2.imshow("Drive Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
