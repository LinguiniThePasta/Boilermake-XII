from pydub import AudioSegment
import simpleaudio as sa

# Load the audio file
audio = AudioSegment.from_file("hell.wav")

# Define a function to play audio from a specific start time
def play_audio_from(timestamp_ms):
    # Extract the segment of audio starting from the given timestamp
    segment = audio[timestamp_ms:]
    # Play the extracted audio segment
    play_obj = sa.play_buffer(
        segment.raw_data,
        num_channels=segment.channels,
        bytes_per_sample=segment.sample_width,
        sample_rate=segment.frame_rate
    )
    return play_obj
