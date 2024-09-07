

import librosa
import os


def check_wav_dimensions(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)  # sr=None to preserve the original sample rate

    # Get the dimensions
    num_channels = 1 if len(audio.shape) == 1 else audio.shape[0]
    num_samples = audio.shape[-1]
    duration = num_samples / sr

    print(f"File: {os.path.basename(file_path)}")
    print(f"Shape: {audio.shape}")
    print(f"Number of channels: {num_channels}")
    print(f"Number of samples: {num_samples}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} seconds")




directory = "speech_n_speech_data/train/track00001"  # Replace with your directory path
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        check_wav_dimensions(file_path)
        print()  # Add a blank line between files