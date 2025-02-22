import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchreid.reid.utils import FeatureExtractor  # Adjusted import based on your package version

# Helper functions
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def cosine_similarity(a, b):
    a_norm = normalize_vector(a)
    b_norm = normalize_vector(b)
    return np.dot(a_norm, b_norm)

# Parameters for matching and updating
SIMILARITY_THRESHOLD = 0.7  # Cosine similarity threshold (closer to 1 means more similar)
ALPHA = 0.5  # Update weight for template averaging

# Load YOLO model for person detection
yolo_model = YOLO('yolov8n.pt')  # Replace with your model path

# Initialize TorchReID feature extractor
extractor = FeatureExtractor(
    model_name='osnet_x0_25',              
    model_path='osnet_x0_25_market1501.pt',  
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Open video capture (e.g., webcam)
cap = cv2.VideoCapture(0)

# Person database: {person_id: feature template}
person_db = {}
next_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect persons in the frame using YOLO
    results = yolo_model(frame)

    for box in results[0].boxes:
        # Get bounding box coordinates, confidence, and class id
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cls = int(box.cls.item())
        # Debug print (optional)
        # print(x1, y1, x2, y2, conf, cls)
        
        # Filter for persons (assuming class 0 is "person")
        if conf > 0.5 and cls == 0:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Crop the person region
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_crop.size > 0:
                # Extract Re-ID features (extractor expects a list of images)
                features = extractor([person_crop])
                feature_vector = features[0]
                if torch.is_tensor(feature_vector):
                    feature_vector = feature_vector.cpu().numpy()
                feature_vector = normalize_vector(feature_vector.flatten())
                
                # Compare with stored templates
                matched_id = None
                for pid, stored_vector in person_db.items():
                    sim = cosine_similarity(feature_vector, stored_vector)
                    # Debug print similarity (optional)
                    print(f"Similarity with ID {pid}: {sim}")
                    if sim > SIMILARITY_THRESHOLD:
                        matched_id = pid
                        # Update the stored template using an exponential moving average
                        updated_vector = ALPHA * stored_vector + (1 - ALPHA) * feature_vector
                        person_db[pid] = normalize_vector(updated_vector)
                        break

                # If no match found, assign a new id and store the template
                if matched_id is None:
                    matched_id = next_id
                    person_db[matched_id] = feature_vector
                    next_id += 1

                # Draw the assigned ID above the bounding box
                cv2.putText(frame, f"ID: {matched_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Person Detection & Re-ID', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

