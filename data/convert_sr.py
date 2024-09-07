import torchaudio
import torch

def convert_sample_rate(input_file, output_file, target_sr=22050):
    # Load the audio file
    waveform, original_sr = torchaudio.load(input_file)

    # Resample if necessary
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(original_sr, target_sr)
        waveform = resampler(waveform)

    # Save the resampled audio
    torchaudio.save(output_file, waveform, target_sr)

# Example usage
input_file = "dummy_speech_n_speech/mixture_2.wav"
output_file = "dummy_speech_n_speech/mixture_2_sampled.wav"
convert_sample_rate(input_file, output_file)