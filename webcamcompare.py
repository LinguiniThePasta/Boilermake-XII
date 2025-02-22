import cv2
import numpy as np
from posecompare import PoseComparator

def capture_webcam_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        return None
    return frame

def main():
    comparator = PoseComparator()
    reference_image = "testdata/cool.jpg"  # Change to the actual reference image path
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        frame = capture_webcam_frame(cap)
        
        similarity = comparator.compare_images(frame, reference_image)
        if similarity is not None:
            text = f"Similarity: {similarity:.4f}"
            color = (0, 255, 0) if similarity < 0.1 else (0, 0, 255)
            
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)       
        cv2.imshow("Webcam", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
