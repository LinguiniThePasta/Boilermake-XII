import pygame

def start_action(bpm_input_box, start_time_input_box, end_time_input_box, link_input_box, screen, width, height, play_video):
    """Handles the start button action by retrieving input values and playing the video."""
    bpm = int(bpm_input_box.get_value())
    start_time = start_time_input_box.get_value()
    end_time = end_time_input_box.get_value()
    youtube_link = link_input_box.get_value()

    print(f"BPM: {bpm}, Start Time: {start_time}, End Time: {end_time}, YouTube Link: {youtube_link}")

    play_video(screen, width, height)
