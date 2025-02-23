import pygame
from start_button_logic import upload_video

def start_action(bpm_input_box, start_time_input_box, song_name_input_box, link_input_box, screen, width, height, play_video):
    """Handles the start button action by retrieving input values and playing the video."""
    bpm = int(bpm_input_box.get_value())
    start_time = int(start_time_input_box.get_value())
    song_name = str(song_name_input_box.get_value())
    youtube_link = str(link_input_box.get_value())

    print(f"BPM: {bpm}, Start Time: {start_time}, Song Name: {song_name}, YouTube Link: {youtube_link}")

    upload_video(bpm, start_time, song_name, youtube_link)

    play_video(screen, width, height, song_name, start_time, bpm)
