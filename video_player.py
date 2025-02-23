import cv2
import pygame
import time
from audio_player import play_audio_from
from shared import *
import math

# Open webcam for face detection
webcam = cv2.VideoCapture(0)  # Change index if using an external webcam

# Load OpenCV's Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face():
    """Detects faces using OpenCV's Haar cascade model and returns face coordinates."""
 
    ret, frame = webcam.read()
    if not ret:
        return False, None  # No frame captured

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    return len(faces) > 0, gray, faces  # Returns True if at least one face is detected


def play_video(screen, width, height, song_name, start_time):
    """Plays video, synchronizes with audio, and overlays a square if a face is detected."""
    cap = cv2.VideoCapture(song_name + '.mp4')
    video_offset = 0.20  # Adjust this offset for better audio-video sync

    # Seek the video to start_time (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    play_audio_from(start_time * 1000)
    real_start_time = time.time()  # Actual time when playback begins
    face_detection_events = []
    prev_time = 0
    prev_beat = 0
    tempo = 120
    while cap.isOpened():
        elapsed_time = time.time() - real_start_time - video_offset + start_time
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
            face_detected, _, _ = detect_face()
            face_detection_events.append((elapsed_time * 1000, face_detected))
            if face_detected:
                print("YES")
                pygame.draw.rect(screen, (255, 0, 0), (width // 2 - 25, height // 2 - 25, 50, 50), 3)  # Red square


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

