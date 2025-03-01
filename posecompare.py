import math
import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cosine
import statistics
import heapq

from get_foreground_people import GetForegroundPersons


class PoseComparator:
    def __init__(self, model_path="yolo11n-pose.pt", device=0):
        self.model = YOLO(model_path)
        self.device = device
        self.foregroundPersons = GetForegroundPersons()
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        return cosine(vec1, vec2)

    @staticmethod
    def is_valid_keypoint(keypoint):
        return not np.all(keypoint == 0)

    @staticmethod
    def l2_normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Avoid division by zero for zero vectors, return as is.
        return vector / norm

    def get_joint_vectors(self, pose):
        left_leg = (pose[15] - pose[13]) if len(pose) > 15 and self.is_valid_keypoint(pose[15]) else None
        right_leg = (pose[16] - pose[14]) if len(pose) > 16 and self.is_valid_keypoint(pose[16]) else None

        vectors = {
            'left_arm_upper': pose[7] - pose[5],
            'left_arm_lower': pose[9] - pose[7],
            'right_arm_upper': pose[8] - pose[6],
            'right_arm_lower': pose[10] - pose[8],
            'left_leg_upper': pose[13] - pose[11],
            'left_leg_lower': left_leg,
            'right_leg_upper': pose[14] - pose[12],
            'right_leg_lower': right_leg,
        }

        normalized_vectors = {}
        for key, vector in vectors.items():
            if vector is not None:
                normalized_vectors[key] = self.l2_normalize(vector)
            else:
                normalized_vectors[key] = None
        return normalized_vectors
    
    def compare_poses(self, pose1, pose2):
        if (len(pose1) < 14 or len(pose2) < 14):
            pass
        pose1_vectors = self.get_joint_vectors(pose1)
        pose2_vectors = self.get_joint_vectors(pose2)

        cosine_similarities = []
        distances = []
        for key in pose1_vectors:
            if pose1_vectors[key] is None or pose2_vectors[key] is None:
                continue
            similarity = self.cosine_similarity(pose1_vectors[key], pose2_vectors[key])
            if (key == 'right_leg_upper' or key == 'left_leg_upper'):
                similarity *= 2
            if (key == 'right_leg_lower' or key == 'left_leg_lower'):
                similarity *= 1.5
            cosine_similarities.append(similarity)
            distances.append(math.sqrt(similarity) * 2)
        
        if len(distances) < 2:
            return 1
        return statistics.mean(distances)
    
    def extract_keypoints(self, results):
        keypoints = []
        for result in results:
            if result.keypoints is not None:
                keypoints.append(result.keypoints.xy[0].cpu().numpy())
        return keypoints

    def analyze_image(self, img_input):
        image = None
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                print(f"Error: Could not read image at {img_input}")
                return None
        elif isinstance(img_input, np.ndarray):
            image = img_input.copy()

        depth_map = self.foregroundPersons.detect_depth(image)
        pose = self.foregroundPersons.extract_people_pose(image)
        filtered_poses = self.foregroundPersons.intersect(depth_map,
                                                          pose,
                                                          image.shape)
        print(filtered_poses)
        return filtered_poses
        # # below takes an image and gets ONE set of keypoints
        # results = self.model(img_path, device=self.device)
        # keypoints = self.extract_keypoints(results)
        # return keypoints[0] if keypoints else None
    
    def reference_to_cam(self, reference_keypoint, web_img):
        cam_pose = self.analyze_image(web_img)
        if reference_keypoint is None or cam_pose is None:
            #print(f"No pose detected in one or both images: {img_path_1}, {img_path_2}")
            return 1
        similarity = self.compare_poses(reference_keypoint, cam_pose)
        if similarity is None:
            similarity = 1
        print(f"Pose similarity between reference and webcam pose is: {similarity}")
        return similarity
    
    def compare_images(self, img_path_1, img_path_2):
        pose1 = self.analyze_image(img_path_1)
        pose2 = self.analyze_image(img_path_2)
        
        if pose1 is None or pose2 is None:
            #print(f"No pose detected in one or both images: {img_path_1}, {img_path_2}")
            return 1


        similarity = self.compare_poses(pose1, pose2)
        print("SIMILARITY")
        print(similarity)

        if similarity is None:
            similarity = 1
        # print(f"Pose similarity between {img_path_1} and {img_path_2}: {similarity}")
        return similarity

# Example usage:
if __name__ == "__main__":
    comparator = PoseComparator()
    comparator.compare_images("testdata/lingyu.jpg", "testdata/pose3.jpg")
    comparator.compare_images("testdata/pose1.jpg", "testdata/pose2.jpg")
    comparator.compare_images("testdata/pose2.jpg", "testdata/pose3.jpg")
