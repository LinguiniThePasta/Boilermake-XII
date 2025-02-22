# ! pip install "yt-dlp[default]"
# ! pip install ultralytics

import yt_dlp
import cv2
import csv
import subprocess
import numpy as np

from ultralytics import YOLO

def upload_video():

    model = YOLO("yolo11n-pose.pt")
    device = "cpu"

    def is_valid_keypoint(keypoint):
        return not np.all(keypoint == 0)

    def l2_normalize(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Avoid division by zero for zero vectors, return as is.
        return vector / norm

    def get_length(filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        return float(result.stdout)

    def extract_keypoints(results):
        keypoints = []
        for result in results:
            if result.keypoints is not None:
                keypoints.append(result.keypoints.xy[0].cpu().numpy())
        return keypoints
        
    def analyze_image(img_path):
        results = model(img_path, device=device)
        keypoints = extract_keypoints(results)
        return keypoints[0] if keypoints else None

    def get_joint_vectors(pose):
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
                normalized_vectors[key] = l2_normalize(vector)
            else:
                normalized_vectors[key] = None
        return normalized_vectors

    # input
    url = "https://www.youtube.com/watch?v=fCv5q2yoqAY"
    songname = "rasputin".replace(" ", "")
    bpm = 128
    start_beat = 25

    # STEP 1:
    # download youtube video

    options = {
        "outtmpl": songname + ".mp4",
        "format": "best"
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    # STEP 2:
    # create csv file with timestamps to sample

    sample_a_frame_every_x_milliseconds = 1000 / (bpm / 60)
    filename = "choreo_to_compare_to.csv"
    fields = ['timestamp', 'visual pose reference', 'pose angles for computational comparison']
    rows = [['0', None, None] for i in range(start_beat * 1000, get_length(songname + ".mp4"), sample_a_frame_every_x_milliseconds)]

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    # STEP 3:
    # get the frame numbers associated with each timestamp
    # process their poses, and store that data in the csv

    choreo_capture = cv2.VideoCapture(songname + ".mp4")
    frame_counter = 0
    current_timestamp_to_look_for = 0 # read csv file to get each value
    while choreo_capture.isOpened():
        frame_exists, curr_frame = choreo_capture.read()
        if frame_exists and choreo_capture.get(cv2.CAP_PROP_POS_MSEC) == current_timestamp_to_look_for:
            cv2.imwrite("frame.jpg", curr_frame)
            # run inference on the frame.jpg and store the pose info in csv
            rows[frame_counter][1] = model.track(curr_frame, device=0, tracker="bytetrack.yaml")[0]
            # run Andrew's encoding and store angle info in csv
            rows[frame_counter][2] = get_joint_vectors("frame.jpg")
