import pygame
from input_boxes import InputBox, Button
from video_player import play_video
import cv2
from dance_stats import display_statistics
from actions import start_action
from shared import *
pygame.init()
pygame.mixer.init()


# Get video dimensions
cap = cv2.VideoCapture('hell.mp4')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Create a Pygame display surface
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Just Dance')

FONT = pygame.font.Font(None, 16)


def main():
    """Display input fields"""
    set_detection_events([(100, "BAD"), (200, "GREAT"), (300, "OK")])
    total_elements = 6
    spacing = height // total_elements - 1

    bpm_input_box = InputBox(width // 2 - 100, spacing * 1 - 40, 200, 30, "BPM")
    start_time_input_box = InputBox(width // 2 - 100, spacing * 2 - 40, 200, 30, "Starting Time")
    song_name_input_box = InputBox(width // 2 - 100, spacing * 3 - 40, 200, 30, "Song Name")
    link_input_box = InputBox(width // 2 - 100, spacing * 4 - 40, 200, 30, "YouTube Link")
    start_button = Button(
        width // 2 - 100, spacing * 5 - 40, 200, 30, "Start",
        lambda: start_action(
            bpm_input_box,
            start_time_input_box,
            song_name_input_box,
            link_input_box,
            screen,
            width,
            height,
            play_video
        )
    )
    stat_button = Button(width // 2 - 100, spacing * 6 - 40, 200, 30, "Stat", display_statistics)

    input_boxes = [
        bpm_input_box, start_time_input_box, song_name_input_box, link_input_box, start_button, stat_button
    ]

    running = True
    while running:
        screen.fill((255, 255, 255))

        for box in input_boxes:
            box.draw(screen)

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            for box in input_boxes:
                box.handle_event(event)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
