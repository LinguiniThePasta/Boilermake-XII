import pygame
from input_boxes import InputBox, Button
from video_player import play_video
import cv2
from dance_stats import display_statistics
from actions import start_action
from shared import *
from loadpage import loadpage
from selectpage import selectpage
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
    total_elements = 4
    spacing = height // total_elements - 1

    bpm_input_box = None
    start_time_input_box = None
    song_name_input_box = None
    link_input_box = None

    song_path = None

    # Wrapper function inside main
    def loadpage_wrapper():
        nonlocal bpm_input_box, start_time_input_box, song_name_input_box, link_input_box
        bpm_input_box, start_time_input_box, song_name_input_box, link_input_box = loadpage(screen, width, height)

    def select_wrapper():
        nonlocal song_path
        print(song_path)
        song_path = selectpage(screen, width, height)
        print(song_path)


    load_button = Button(
        width // 2 - 100, spacing * 1 - 40, 200, 30, "Upload",
        loadpage_wrapper
    )
    select_button = Button(
        width // 2 - 100, spacing * 2 - 40, 200, 30, "Select",
        select_wrapper
    )
    start_button = Button(
        width // 2 - 100, spacing * 3 - 40, 200, 30, "Start",
        lambda: start_action(
            song_path,
            screen,
            width,
            height,
            play_video
        )
    )
    stat_button = Button(width // 2 - 100, spacing * 4 - 40, 200, 30, "Stat", display_statistics)

    input_boxes = [
        load_button, select_button, start_button, stat_button
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
