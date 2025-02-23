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
        "outtmpl": songname + ".mp4",
        "format": "best"
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    # STEP 2:
    # create csv file with timestamps to sample

    sample_a_frame_every_x_milliseconds = int(1000 / int(bpm / 60))

    filename = "choreo_to_compare_to.csv"
    fields = ['timestamp', 'visual pose reference', 'pose angles for computational comparison']
    rows = [[i, None, None] for i in
            range(start_beat * 1000, int(get_length(songname + ".mp4") * 1000), sample_a_frame_every_x_milliseconds)]

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
