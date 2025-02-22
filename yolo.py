import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Variables to calculate FPS
prev_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform inference
    results = model(frame, device='cpu')

    # Get current time
    current_time = time.time()

    # Calculate FPS
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS on frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    annotated_frame = results[0].plot()
    # Display the resulting frame
    cv2.imshow('YOLO Inference', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
