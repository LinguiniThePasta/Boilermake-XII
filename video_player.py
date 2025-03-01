import cv2
import csv
import pygame
import time
import numpy as np
from get_foreground_people import GetForegroundPersons
from shared import *
import math
from posecompare import PoseComparator
from moviepy import *
import threading
import os


# Open webcam for face detection
webcam = cv2.VideoCapture(0)  # Change index if using an external webcam
foreground_detector = GetForegroundPersons()

pose_comparator = PoseComparator()
reference_image = "testdata/lingyu.jpg"


# Extract audio from mp4
def extract_audio(video_filename):
    # Load the video file
    video = VideoFileClip(video_filename)

    # Extract and save the audio
    audio_filename = video_filename.replace('.mp4', '.wav')
    video.audio.write_audiofile(audio_filename)
    return audio_filename

def display_feedback(screen, width, height, score_result, effect_start_time):
    """Displays a fading border based on the score result."""
    elapsed_effect_time = (time.time() - effect_start_time) * 1000  # Convert to ms
    effect_duration = 500  # Effect lasts 500 millisecondsss

    if elapsed_effect_time > effect_duration:
        return  # Don't draw if the effect has expired

    # Determine color based on score
    color_map = {
        "GREAT": (0, 255, 0),  # Green
        "OK": (255, 255, 0),  # Yellow
        "BAD": (255, 0, 0)  # Red
    }
    color = color_map.get(score_result, (0, 0, 0))  # Default to white if unknown

    # Calculate fade-out effect (opacity decrease)
    fade_factor = max(0, 1 - (elapsed_effect_time / effect_duration))
    alpha = int(255 * fade_factor)

    # Draw fading border
    border_thickness = 10
    border_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(border_surface, (*color, alpha), (0, 0, width, height), border_thickness)
    screen.blit(border_surface, (0, 0))

def score(pose_keypoints):
    """Detects faces using OpenCV's Haar cascade model and returns face coordinates."""
    ret, frame = webcam.read()
    similarity = pose_comparator.reference_to_cam(pose_keypoints, frame)
    print(similarity)
    if similarity is None:
        return "BAD"
    if (similarity < 0.6):
        return "GREAT"
    elif (similarity < 0.9):
        return "OK"
    else:
        return "BAD"

def get_tempo(folderpath):
    folder_name = os.path.basename(folderpath)
    metafile = os.path.join(folderpath, f"{folder_name}.meta")
    
    # Check if the metafile exists
    if not os.path.exists(metafile):
        print(f"Error: {metafile} not found.")
        return None
    
    # Read the number from the file
    try:
        with open(metafile, "r") as file:
            number = int(file.read().strip())  # Convert the content to an integer
            return number
    except (ValueError, FileNotFoundError) as e:
        print(f"Error reading {metafile}: {e}")
        return None
    
    print(f"Read number from {metafile}: {number}")

def play_video(folderpath, screen, width, height):
    folder_name = os.path.basename(folderpath)
    video_filename = folderpath + "\\" + folder_name + ".mp4"
    audio_filename = extract_audio(video_filename)
    pygame.mixer.music.load(audio_filename)
    pygame.mixer.music.set_volume(1)
    pygame.mixer.music.play()
    tempo = get_tempo(folderpath)
    print(tempo)
    """Plays video, synchronizes with audio, and overlays a square if a face is detected."""

    cap = cv2.VideoCapture(folderpath + "\\" + folder_name  + ".mp4")
    song_name = folder_name
    video_offset = 0.20  # Adjust this offset for better audio-video sync

    # Seek the video to start_time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)

    face_detection_events = []
    prev_time = 0
    prev_beat = 0
    effect_start_time = None  # Track when the effect starts
    score_result = None  # Track the latest score
    real_start_time = time.time()
    timestamps_and_poses = get_huge_shit(folder_name)
    print(timestamps_and_poses)

    begin_song_time = int(timestamps_and_poses[0][0])
    print(begin_song_time)
    while cap.isOpened():
        elapsed_time = time.time() - real_start_time - video_offset
        elapsed_time = max(0.001, elapsed_time)
        cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_time * 1000)
        ret, frame = cap.read()

        if not ret:
            break


        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        # Create a Pygame surface from the frame
        frame_surface = pygame.surfarray.make_surface(frame)
        frame_surface = pygame.transform.rotate(frame_surface, -90)
        frame_surface = pygame.transform.flip(frame_surface, True, False)

        # Blit the frame to the screen
        screen.blit(frame_surface, (0, 0))
        time_diff = elapsed_time - prev_time
        fps = "N/A"
        if time_diff > 0:
            fps = str(round(1.0 / time_diff))

        prev_time = elapsed_time
        fps_font = pygame.font.SysFont("Arial", 18)
        fps_text = fps_font.render(fps, True, pygame.Color("white"))
        screen.blit(fps_text, (10, 10))  # Position the FPS at the top-left corner
        # Detect face and draw a square if detected

        beats_per_second = tempo / 60
        current_beat = max(0, math.floor((elapsed_time - begin_song_time*1000) * beats_per_second)) # Beat count based on elapsed time
        print("CURRENT BEAT IS" + str(current_beat))
        # Check if the beat has changed (i.e., if it's greater than the previous stored beat)
        if current_beat > prev_beat:
            prev_beat = current_beat  # Update the stored beat value
            print("BEAT")

            def process_beat():
                nonlocal face_detection_events, effect_start_time, score_result
                if (current_beat >= len(timestamps_and_poses)):
                    return
                current_pose = timestamps_and_poses[current_beat][1]
                score_result = score(current_pose)  # Score result could be BAD, GOOD, or GREAT
                face_detection_events.append((elapsed_time * 1000, score_result))
                effect_start_time = time.time()

                print("Processed BEAT on separate thread")

            process_beat()
            # Start a new thread for processing the score
            # threading.Thread(target=process_beat).start()

        if effect_start_time:
            display_feedback(screen, width, height, score_result, effect_start_time)
        pygame.display.update()

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                webcam.release()
                pygame.quit()
                return

    cap.release()
    #webcam.release()
    print(face_detection_events)
    set_detection_events(face_detection_events)
    return

