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
    def __init__(self, device=0):
        self.device = device
        self.foregroundPersons = GetForegroundPersons()

    JOINT_WEIGHTS = {
        'left_arm_upper': 1.0,
        'left_arm_lower': 1.0,
        'right_arm_upper': 1.0,
        'right_arm_lower': 1.0,
        'left_leg_upper': 2.0,
        'left_leg_lower': 1.5,
        'right_leg_upper': 2.0,
        'right_leg_lower': 1.5,
        'shoulder_line': 1.2,
        'hip_line': 1.2,
        'torso': 1.5,
    }

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
            return None
        return vector / norm

    def get_joint_vectors(self, pose):
        def safe_vec(a_idx, b_idx):
            if len(pose) <= max(a_idx, b_idx):
                return None
            if not PoseComparator.is_valid_keypoint(pose[a_idx]) or not PoseComparator.is_valid_keypoint(pose[b_idx]):
                return None
            return PoseComparator.l2_normalize(pose[a_idx] - pose[b_idx])

        return {
            'left_arm_upper': safe_vec(7, 5),
            'left_arm_lower': safe_vec(9, 7),
            'right_arm_upper': safe_vec(8, 6),
            'right_arm_lower': safe_vec(10, 8),
            'left_leg_upper': safe_vec(13, 11),
            'left_leg_lower': safe_vec(15, 13),
            'right_leg_upper': safe_vec(14, 12),
            'right_leg_lower': safe_vec(16, 14),
            'shoulder_line': safe_vec(6, 5),
            'hip_line': safe_vec(12, 11),
            'torso': safe_vec(11, 5),
        }

    def compare_poses(self, pose1, pose2):
        if (len(pose1) < 14 or len(pose2) < 14):
            return 1
        pose1_vectors = self.get_joint_vectors(pose1)
        pose2_vectors = self.get_joint_vectors(pose2)

        distances = []
        total_weight = 0
        for key in pose1_vectors:
            v1, v2 = pose1_vectors[key], pose2_vectors[key]
            weight = self.JOINT_WEIGHTS.get(key, 1.0)


            if v1 is None or v2 is None:
                # At least one joint not visible — skip this joint entirely
                continue
            else:
                similarity = self.cosine_similarity(v1, v2)
                distances.append(weight * math.sqrt(similarity) * 2)
            total_weight += weight

        if total_weight == 0 or len(distances) < 2:
            return 1.0

        return sum(distances) / total_weight

    def analyze_image(self, img_input):
        if isinstance(img_input, str):
            image = cv2.imread(img_input)
            if image is None:
                print(f"Error: Could not read image at {img_input}")
                return None
        elif isinstance(img_input, np.ndarray):
            image = img_input.copy()
        else:
            return []
        depth_map = self.foregroundPersons.detect_depth(image)
        pose = self.foregroundPersons.extract_people_pose(image)
        filtered_poses = self.foregroundPersons.intersect(depth_map,
                                                          pose,
                                                          image.shape)
        return filtered_poses

    def compare_all_players(self, reference_keypoints, web_img):

        if reference_keypoints is None:
            return {}

        people = self.analyze_image(web_img)
        if not people:
            return {}

        scores = {}
        for (track_id, keypoints) in people:
            scores[track_id] = self.compare_poses(reference_keypoints, keypoints)

        return scores

    def compare_images(self, img_path_1, img_path_2):
        # this is for testing purposes, basically just extracts one person and tries to compare that
        people1 = self.analyze_image(img_path_1)
        people2 = self.analyze_image(img_path_2)

        if not people1 or not people2:
            return 1.0

        # Use the first detected foreground person from each image
        pose1 = people1[0][1]
        pose2 = people2[0][1]

        similarity = self.compare_poses(pose1, pose2)
        print(f"Similarity: {similarity:.4f}")
        return similarity


# Example usage:
if __name__ == "__main__":
    comparator = PoseComparator()
    comparator.compare_images("testdata/lingyu.jpg", "testdata/pose3.jpg")
    comparator.compare_images("testdata/pose1.jpg", "testdata/pose2.jpg")
    comparator.compare_images("testdata/pose2.jpg", "testdata/pose3.jpg")
