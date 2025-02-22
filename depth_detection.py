from ultralytics import YOLO
import torch
import torchvision
import sys
import os
import cv2
import time
import numpy as np

# --- Path Configuration for Depth Anything ---
script_directory = os.path.dirname(os.path.abspath(__file__))
depth_anything_root = os.path.join(script_directory, 'Depth-Anything-V2')
sys.path.append(depth_anything_root)
from depth_anything_v2.dpt import DepthAnythingV2

# --- Device Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
elif DEVICE == 'mps':
    print(f"MPS is available: {torch.backends.mps.is_available()}")

# --- Model Configuration ---
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vits'

# --- Load Depth Anything Model ---
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- FPS Variables ---
prev_time = 0
frame_count = 0

# --- Window Setup ---
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

# --- Main Loop ---
while frame_count <= 300:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame_rgb = cv2.resize(frame_rgb, (1080, 1920))

    depth = model.infer_image(resized_frame_rgb)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # --- Display FPS on the DEPTH MAP FRAME (depth_colored) ---
    cv2.putText(depth_colored, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # <-- Changed to depth_colored

    cv2.imshow('Depth Map', depth_colored) # Show depth map with FPS

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()