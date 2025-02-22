# ! python3 -m pip install -U "yt-dlp[default]"

import yt_dlp
import cv2
import csv
import subprocess

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

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
        # run inference on the frame.jpg to get the pose
        # store the pose info in csv
        # run Andrew's encoding
        # store angle info in csv
