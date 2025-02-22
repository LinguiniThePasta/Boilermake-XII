import cv2
import torch
from strongsort.strong_sort import StrongSORT
from ultralytics import YOLO

# Load YOLO-Pose model (running on CPU)
model_path = "yolo11n-pose.pt"  # Make sure this file is in your directory
device = "cpu"  # Force CPU usage
model = YOLO("yolo11n-pose.pt")

# Initialize DeepSORT tracker
strongsort = StrongSORT(model_weights="osnet_x0_25_market1501.pt", device=device)

cap = cv2.VideoCapture(0)  # Change to 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO-Pose detection
    results = model(frame, device=device)
    
    detections = []
    for result in results:
        for box in result.boxes:  # Loop through detected objects
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get bounding box
            conf = box.conf[0].item()  # Get confidence score
            class_id = int(box.cls[0].item())  # Get class ID
            
            if class_id == 0:  # Class ID 0 = Person
                detections.append(([x1, y1, x2, y2], conf, None))  # No class embedding needed


    tracked_objects = strongsort.update(detections, frame)

    # Draw results
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = track.to_ltrb()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
