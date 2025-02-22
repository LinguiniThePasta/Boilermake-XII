import cv2
import pygame
import time
from audio_player import play_audio_from

# Open the video file using OpenCV
cap = cv2.VideoCapture('hell.mp4')

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

    return len(faces) > 0, frame, faces  # Returns True if at least one face is detected

def play_video(screen, width, height):
    """Plays video, synchronizes with audio, and overlays a square if a face is detected."""
    video_offset = -0.05  # Adjust this offset for better audio-video sync
    start_time = time.time()
    play_audio_from((time.time() - start_time) * 1000)
    prev_time = 0
    while cap.isOpened():
        elapsed_time = time.time() - start_time - video_offset
        elapsed_time = max(0, elapsed_time)
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
        fps = str(round(1.0 / (elapsed_time - prev_time)))
        prev_time = elapsed_time
        fps_font = pygame.font.SysFont("Arial", 18)
        fps_text = fps_font.render(fps, True, pygame.Color("white"))
        screen.blit(fps_text, (10, 10))  # Position the FPS at the top-left corner
        # Detect face and draw a square if detected
        face_detected, _, _ = detect_face()
        if face_detected:
            pygame.draw.rect(screen, (255, 0, 0), (width//2 - 25, height//2 - 25, 50, 50), 3)  # Red square

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
    pygame.quit()
