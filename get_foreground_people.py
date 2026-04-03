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

        # depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        # print(f"detect_depth: depth_resized shape: {depth_resized.shape}")
        # print(f"detect_depth: depth_colored shape: {depth_colored.shape}")
        return depth_resized

    def extract_people_pose(self, frame):
        results = self.yolo_pose_model.track(source=frame, classes=0,
                                             verbose=False, persist=True,
                                             tracker="bytetrack.yaml")
        return results

    def intersect(self, depth_map, pose_results, frame_shape):
        fg_threshold = np.percentile(depth_map, 60)

        people = []
        if not pose_results or not pose_results[0].keypoints or not pose_results[0].boxes:
            return np.array([])

        for person_result in pose_results:
            boxes = person_result.boxes
            if boxes is None or boxes.id is None:
                continue

            track_ids = boxes.id.int().cpu().tolist()
            keypoints_all = person_result.keypoints.xy.cpu().numpy()

            # Get x and y coordinates and convert to integers
            for i, (track_id, keypoints) in enumerate(zip(track_ids, keypoints_all)):
                x_coords = np.clip(keypoints[:, 0].astype(int), 0, depth_map.shape[1] - 1)
                y_coords = np.clip(keypoints[:, 1].astype(int), 0, depth_map.shape[0] - 1)
                avg_depth = np.mean(depth_map[y_coords, x_coords])
                if avg_depth >= fg_threshold:
                    people.append((track_id, keypoints))
        return people



    def run(self, frame_num = 10000):
        prev_time = 0
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
        frame_count = 0
        while frame_count <= frame_num:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            depth_resized = self.detect_depth(frame)
            pose_results = self.extract_people_pose(frame)
            filtered_poses = self.intersect(depth_resized, pose_results, frame.shape)

            depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

            print(filtered_poses)

            annotated_frame = frame.copy()

            # Plot Filtered Poses
            # if len(filtered_poses) > 0:
            #     for person_pose in filtered_poses:
            #         person_pose_int = np.round(person_pose).astype(int)
            #         for kp in person_pose_int:
            #             print(kp)
                        # x, y = kp[0], kp[1]
                        # cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('YOLO Inference', annotated_frame)
            cv2.imshow('Depth Map', depth_colored)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def extract_last_pose(self, grace_period = 70):
        frame_count = 0
        filtered_poses = None
        while frame_count <= grace_period:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            depth_colored, depth_raw = self.detect_depth(frame)
            pose_results = self.extract_people_pose(frame)
            filtered_poses = self.intersect(depth_raw, pose_results, frame.shape)

        self.cap.release()
        return filtered_poses

if __name__ == "__main__":
    foreground_detector = GetForegroundPersons()
    foreground_detector.run()
