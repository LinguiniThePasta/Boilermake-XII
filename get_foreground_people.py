import cv2
from ultralytics import YOLO
import torch
import sys
import os
import time
import numpy as np

script_directory = os.path.dirname(os.path.abspath(__file__))
depth_anything_root = os.path.join(script_directory, 'Depth-Anything-V2')
sys.path.append(depth_anything_root)
from depth_anything_v2.dpt import DepthAnythingV2

class GetForegroundPersons():
    def __init__(self, source=0):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.depth_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.encoder = 'vits'

        # Load Depth Anything Model
        self.depth_model = DepthAnythingV2(**self.depth_model_configs[self.encoder])
        self.depth_model.load_state_dict(
            torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cpu'))
        self.depth_model = self.depth_model.to(self.DEVICE).eval()

        # Load YOLO Pose Model
        self.yolo_pose_model = YOLO('yolo11n-pose.pt')
        self.yolo_pose_model.classes = [0]
        self.yolo_pose_model.to(self.DEVICE)

        # Load webcam
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Misc
        self.expand = 20

    def detect_depth(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame_rgb = cv2.resize(frame_rgb, (320, 240))

        depth = self.depth_model.infer_image(resized_frame_rgb)
        depth_resized = cv2.resize(depth, (frame.shape[1], frame.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)

        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        return depth_colored, depth_resized

    def extract_people_pose(self, frame):
        results = self.yolo_pose_model.predict(source=frame, classes=0, verbose=False)
        return results

    def intersect(self, depth_map, pose_results, frame_shape): # Added frame_shape
        filtered_poses = []
        if not pose_results or not pose_results[0].keypoints or not pose_results[0].boxes:
            return np.array([])

        boxes_np = pose_results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes_np):
            x_min, y_min, x_max, y_max = map(int, box)

            # Clip box coordinates
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame_shape[1], x_max) # Use frame_shape for clipping
            y_max = min(frame_shape[0], y_max)

            box_depth_values = depth_map[y_min:y_max, x_min:x_max].flatten()

            if box_depth_values.size == 0:
                continue

            mean_box_depth = np.mean(box_depth_values) if box_depth_values.size > 0 else float('inf') # Initialize to infinity if empty

            # Define background region (expand box by certain number of pixels in each direction)
            bg_x_min = max(0, x_min - self.expand)
            bg_y_min = max(0, y_min - self.expand)
            bg_x_max = min(frame_shape[1], x_max + self.expand) # Use frame_shape for clipping
            bg_y_max = min(frame_shape[0], y_max + self.expand)

            # Exclude the person's box region from the background region
            bg_mask = np.ones(depth_map.shape[:2], dtype=bool)
            bg_mask[y_min:y_max, x_min:x_max] = False # Mask out the person's box
            bg_depth_values = depth_map[bg_y_min:bg_y_max, bg_x_min:bg_x_max][bg_mask[bg_y_min:bg_y_max, bg_x_min:bg_x_max]].flatten()


            mean_bg_depth = np.mean(bg_depth_values) if bg_depth_values.size > 0 else float('inf') # Initialize to infinity if empty

            print(f"Box {i}: Mean Box Depth: {mean_box_depth:.2f}, Mean BG Depth: {mean_bg_depth:.2f}")
            depth_ratio = 0
            if mean_bg_depth != float('inf') and mean_bg_depth != 0:
                depth_ratio = mean_box_depth / mean_bg_depth
            print(f"Box {i}: Depth Ratio (Box/BG): {depth_ratio:.2f}")

            if mean_box_depth != float('inf') and mean_bg_depth != float(
                    'inf') and mean_box_depth >= 2 * mean_bg_depth:
                print(f"Box {i}: Foreground - Condition met (Box Depth < 0.1 * BG Depth)")
                filtered_poses.append(pose_results[0].keypoints.xy.cpu().numpy()[i])
            else:
                print(f"Box {i}: Background - Condition NOT met (Box Depth >= 0.1 * BG Depth)")
        return np.array(filtered_poses)


    def run(self):
        prev_time = 0
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            depth_colored, depth_raw = self.detect_depth(frame)
            pose_results = self.extract_people_pose(frame)
            filtered_poses = self.intersect(depth_raw, pose_results, frame.shape)

            annotated_frame = frame.copy()

            # Plot Filtered Poses
            if len(filtered_poses) > 0:
                for person_pose in filtered_poses:
                    person_pose_int = np.round(person_pose).astype(int)
                    for kp in person_pose_int:
                        cv2.circle(annotated_frame, tuple(kp), 5, (0, 255, 0), -1)


            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('YOLO Inference', annotated_frame)
            cv2.imshow('Depth Map', depth_colored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    foreground_detector = GetForegroundPersons()
    foreground_detector.run()
