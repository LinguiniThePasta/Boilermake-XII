import cv2
import pygame
import time
from audio_player import play_audio_from
from shared import *
import math
from posecompare import PoseComparator

# Open the video file using OpenCV
cap = cv2.VideoCapture('hell.mp4')

# Open webcam for face detection
webcam = cv2.VideoCapture(0)  # Change index if using an external webcam

# Load OpenCV's Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

pose_comparator = PoseComparator()
reference_image = "testdata/lingyu.jpg"



def display_feedback(screen, width, height, score_result, effect_start_time):
    """Displays a fading border based on the score result."""
    elapsed_effect_time = (time.time() - effect_start_time) * 1000  # Convert to ms
    effect_duration = 500  # Effect lasts 10 milliseconds

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

def score():
    """Detects faces using OpenCV's Haar cascade model and returns face coordinates."""
 
    ret, frame = webcam.read()
    similarity = pose_comparator.compare_images(frame, reference_image)
    if similarity is None:
        return "BAD"
    if (similarity < 0.25):
        return "GREAT"
    elif (similarity < 0.35):
        return "OK"
    else:
        return "BAD"

def play_video(screen, width, height):

    """Plays video, synchronizes with audio, and overlays a square if a face is detected."""
    video_offset = 0.20  # Adjust this offset for better audio-video sync
    start_time = time.time()
    play_audio_from((time.time() - start_time) * 1000)
    face_detection_events = []
    prev_time = 0
    prev_beat = 0
    tempo = 60


    effect_start_time = None  # Track when the effect starts
    score_result = None  # Track the latest score
    while cap.isOpened():
        elapsed_time = time.time() - start_time - video_offset
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
        current_beat = math.floor(elapsed_time * beats_per_second)  # Beat count based on elapsed time
        # Check if the beat has changed (i.e., if it's greater than the previous stored beat)
        if current_beat > prev_beat:
            prev_beat = current_beat  # Update the stored beat value
            print("BEAT")
            # Run face detection only if the beat has changed
            score_result = score()
            face_detection_events.append((elapsed_time * 1000, score_result))
            effect_start_time = time.time()
            #score result could be BAD, GOOD, or GREAT

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
    webcam.release()
    print(face_detection_events)
    set_detection_events(face_detection_events)
    return

