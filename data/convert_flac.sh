#!/bin/bash
# Convert all FLAC files in librispeech_aud1 to WAV
for f in librispeech_aud1/*.flac; do
  ffmpeg -i "$f" "${f%.flac}.wav"
  rm "$f" # Remove the original .flac file
done
