# ! pip install "yt-dlp[default]"
# ! pip install ultralytics

import yt_dlp
import cv2
import csv
import subprocess
import numpy as np
from ultralytics import YOLO


def upload_video(bpm, start_beat, songname, url):
    model = YOLO("yolo11n-pose.pt")
    device = "cpu"

    def get_length(input_video):
        video_capture = cv2.VideoCapture(input_video)

        if not video_capture.isOpened():
            return -1

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            return -1

        duration = int(frame_count / fps)
        return duration

    # STEP 1:
    # download youtube video

    options = {
        "outtmpl": "/song/" + songname + "/" + songname + ".mp4",
        "format": "best"
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    # STEP 2:
    # create csv file with timestamps to sample

    sample_a_frame_every_x_milliseconds = int(1000 / int(bpm / 60))

    filename = "/song/" + songname + "/" + songname + .csv"
    fields = ['timestamp', 'visual pose reference']
    rows = [[i, None] for i in
            range(start_beat * 1000, int(get_length(songname + ".mp4") * 1000), sample_a_frame_every_x_milliseconds)]

    # STEP 3:
    # gathering poses from video

    choreo_capture = cv2.VideoCapture("/song/" + songname + "/" + songname + ".mp4")
    csv_rows_checked_off = 0
    current_csv_row_timestamp_to_look_for = rows[csv_rows_checked_off][0]

    fps = choreo_capture.get(cv2.CAP_PROP_FPS)
    count_total_frames = 0

    while choreo_capture.isOpened() and csv_rows_checked_off < (len(rows) - 1):
        frame_exists, curr_frame = choreo_capture.read()
        count_total_frames += 1
        timestamp = int(1000 * count_total_frames / fps)
        if frame_exists:
            if timestamp >= current_csv_row_timestamp_to_look_for:
                cv2.imwrite("frame.jpg", curr_frame)
                csv_rows_checked_off += 1
                current_csv_row_timestamp_to_look_for = rows[csv_rows_checked_off][0]
                results = model("frame.jpg")
                rows[csv_rows_checked_off][1] = results[0].keypoints.xy[0].cpu().numpy()

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
