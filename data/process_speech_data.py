import os
import librosa
import soundfile as sf
import shutil
import random


def normalize_duration(file_path, target_duration=10.0, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    length = len(audio) / sr
    if length > target_duration:
        # Trim audio
        audio = audio[:int(target_duration * sr)]
    else:
        # Pad audio
        audio = librosa.util.fix_length(audio, size=int(target_duration * sr))
    return audio


def process_directory(directory, output_directory, target_duration=20.0, sr=22050):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    processed_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio = normalize_duration(file_path, target_duration, sr)
                output_path = os.path.join(output_directory, f"Track{len(processed_files):05d}.wav")
                sf.write(output_path, audio, sr)
                processed_files.append(output_path)

    return processed_files


def equalize_file_count(dir1, dir2):
    files1 = [f for f in os.listdir(dir1) if f.endswith('.wav')]
    files2 = [f for f in os.listdir(dir2) if f.endswith('.wav')]

    if len(files1) < len(files2):
        smaller_dir, larger_count = dir1, len(files2)
        smaller_files = files1
    else:
        smaller_dir, larger_count = dir2, len(files1)
        smaller_files = files2

    while len(smaller_files) < larger_count:
        file_to_duplicate = random.choice(smaller_files)
        new_file_name = f"Track{len(smaller_files):05d}.wav"
        shutil.copy(os.path.join(smaller_dir, file_to_duplicate),
                    os.path.join(smaller_dir, new_file_name))
        smaller_files.append(new_file_name)

    print(f"Equalized file count. Both directories now have {larger_count} files.")


def split_data(directory, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1):
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    # random.shuffle(files)

    train_end = int(len(files) * train_ratio)
    val_end = int(len(files) * (train_ratio + val_ratio))

    return {
        'train': files[:train_end],
        'validation': files[train_end:val_end],
        'test': files[val_end:]
    }


def organize_files(source_dir, target_dir, split_dict):
    for split, files in split_dict.items():
        split_dir = os.path.join(target_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for file in files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(split_dir, file))


# Main execution
if __name__ == "__main__":
    # Process speech data
    speech_processed = process_directory('librispeech_aud1', 'speech_22050_temp')

    # Process non-speech data
    non_speech_processed = process_directory('FSDnoisy18k.audio', 'non_speech_22050_temp')

    # Equalize file count
    equalize_file_count('speech_22050_temp', 'non_speech_22050_temp')

    # Split data
    speech_split = split_data('speech_22050_temp')
    non_speech_split = split_data('non_speech_22050_temp')

    # Organize files
    organize_files('speech_22050_temp', 'speech_22050_2', speech_split)
    organize_files('non_speech_22050_temp', 'non_speech_22050_2', non_speech_split)

    # Clean up temporary directories
    shutil.rmtree('speech_22050_temp')
    shutil.rmtree('non_speech_22050_temp')

    print("Processing complete. Check speech_22050_1 and non_speech_22050_1 directories.")