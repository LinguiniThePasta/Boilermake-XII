# ! pip install "yt-dlp[default]"
# ! pip install ultralytics

import yt_dlp
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from shared import *


def upload_video(bpm, start_beat, songname, url):
    model = YOLO("yolo11n-pose.pt")

    song_dir = Path("./song") / songname
    song_dir.mkdir(parents=True, exist_ok=True)
    video_path = song_dir / f"{songname}.mp4"
    meta_path  = song_dir / f"{songname}.meta"

    # STEP 1: download youtube video
    options = {
        "outtmpl": str(video_path),
        "format": "best"
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    # STEP 2: build timestamp list (one entry per beat)
    def get_length(path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return -1
        fps = cap.get(cv2.CAP_PROP_FPS)
        fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or fc <= 0:
            return -1
        return int(fc / fps)

    sample_ms = int(1000 / (bpm / 60))
    duration_ms = get_length(video_path) * 1000
    rows = [[i, None] for i in range(start_beat * 1000, duration_ms, sample_ms)]

    # STEP 3: extract a pose keyframe at each beat timestamp
    cap = cv2.VideoCapture(str(video_path))
    csv_idx = 0
    total_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened() and csv_idx < len(rows) - 1:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        timestamp = int(1000 * total_frames / fps)
        if timestamp >= rows[csv_idx][0]:
            csv_idx += 1
            cv2.imwrite("frame.jpg", frame)
            results = model("frame.jpg")
            # Guard against frames where nobody is detected
            kp = results[0].keypoints
            if kp is not None and len(kp.xy) > 0:
                rows[csv_idx][1] = kp.xy[0].cpu().numpy()

    cap.release()

    add_huge_shit(songname, rows)
    print(get_huge_shit(songname))

    with open(meta_path, 'w') as metafile:
        metafile.write(str(bpm))
