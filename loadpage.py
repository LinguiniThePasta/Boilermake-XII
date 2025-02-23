import pygame
from input_boxes import InputBox, Button
from actions import load_action

running = True
def loadwrapper(bpm_input_box, start_time_input_box, song_name_input_box, link_input_box):
    global running
    load_action(bpm_input_box, start_time_input_box, song_name_input_box, link_input_box)
    running = False
    print(running)

def loadpage(screen, width, height):
    global running
    total_elements = 5
    spacing = height // total_elements - 1
    bpm_input_box = InputBox(width // 2 - 100, spacing * 1 - 40, 200, 30, "BPM")
    start_time_input_box = InputBox(width // 2 - 100, spacing * 2 - 40, 200, 30, "Starting Time")
    song_name_input_box = InputBox(width // 2 - 100, spacing * 3 - 40, 200, 30, "Song Name")
    link_input_box = InputBox(width // 2 - 100, spacing * 4 - 40, 200, 30, "YouTube Link")
    
    load_button = Button(
        width // 2 - 100, spacing * 5 - 40, 200, 30, "Upload!",
        lambda: loadwrapper(
            bpm_input_box,
            start_time_input_box,
            song_name_input_box,
            link_input_box
        )
    )
    input_boxes = [
        bpm_input_box, start_time_input_box, song_name_input_box, link_input_box, load_button
    ]

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
    
    return bpm_input_box, start_time_input_box, song_name_input_box, link_input_box