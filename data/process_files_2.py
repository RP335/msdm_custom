import os
import librosa
import soundfile as sf
import shutil
import random


def get_file_size(file_path):
    return os.path.getsize(file_path)


def normalize_duration(file_path, target_duration, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    length = len(audio) / sr
    if length > target_duration:
        # Trim audio
        audio = audio[:int(target_duration * sr)]
    else:
        # Pad audio
        audio = librosa.util.fix_length(audio, size=int(target_duration * sr))
    return audio


def equalize_file_count(dir1, dir2):
    files1 = [f for f in os.listdir(dir1) if f.endswith('.wav')]
    files2 = [f for f in os.listdir(dir2) if f.endswith('.wav')]

    if len(files1) < len(files2):
        smaller_dir, larger_dir = dir1, dir2
        smaller_files, larger_files = files1, files2
    else:
        smaller_dir, larger_dir = dir2, dir1
        smaller_files, larger_files = files2, files1

    while len(smaller_files) < len(larger_files):
        file_to_duplicate = random.choice(smaller_files)
        new_file_name = f"Track{len(smaller_files):05d}.wav"
        shutil.copy(os.path.join(smaller_dir, file_to_duplicate),
                    os.path.join(smaller_dir, new_file_name))
        smaller_files.append(new_file_name)

    print(f"Equalized file count. Both directories now have {len(larger_files)} files.")


def sort_files_by_size(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    return sorted(files, key=lambda x: get_file_size(os.path.join(directory, x)))


def normalize_paired_files(speech_dir, non_speech_dir, output_speech_dir, output_non_speech_dir):
    speech_files = sort_files_by_size(speech_dir)
    non_speech_files = sort_files_by_size(non_speech_dir)

    for i, (speech_file, non_speech_file) in enumerate(zip(speech_files, non_speech_files)):
        speech_path = os.path.join(speech_dir, speech_file)
        non_speech_path = os.path.join(non_speech_dir, non_speech_file)

        speech_duration = librosa.get_duration(filename=speech_path)
        non_speech_duration = librosa.get_duration(filename=non_speech_path)

        target_duration = max(speech_duration, non_speech_duration)

        normalized_speech = normalize_duration(speech_path, target_duration)
        normalized_non_speech = normalize_duration(non_speech_path, target_duration)

        sf.write(os.path.join(output_speech_dir, f"Track{i:05d}.wav"), normalized_speech, 22050)
        sf.write(os.path.join(output_non_speech_dir, f"Track{i:05d}.wav"), normalized_non_speech, 22050)


def split_data(directory, seed=42, train_ratio=0.75, val_ratio=0.15, test_ratio=0.1):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])

    random.seed(seed)

    indices = list(range(len(files)))
    random.shuffle(indices)

    train_end = int(len(files) * train_ratio)
    val_end = int(len(files) * (train_ratio + val_ratio))

    return {
        'train': [files[i] for i in indices[:train_end]],
        'validation': [files[i] for i in indices[train_end:val_end]],
        'test': [files[i] for i in indices[val_end:]]
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
    speech_dir = 'librispeech_aud1'
    non_speech_dir = 'FSDnoisy18k.audio'

    # Step 1: Equalize file count
    equalize_file_count(speech_dir, non_speech_dir)

    # Step 2 & 3: Sort files by size and normalize paired files
    temp_speech_dir = 'temp_speech'
    temp_non_speech_dir = 'temp_non_speech'
    os.makedirs(temp_speech_dir, exist_ok=True)
    os.makedirs(temp_non_speech_dir, exist_ok=True)

    normalize_paired_files(speech_dir, non_speech_dir, temp_speech_dir, temp_non_speech_dir)

    # Step 4: Shuffle and split data
    speech_split = split_data(temp_speech_dir)
    non_speech_split = split_data(temp_non_speech_dir)

    # Organize files
    organize_files(temp_speech_dir, 'speech_22050_2', speech_split)
    organize_files(temp_non_speech_dir, 'non_speech_22050_2', non_speech_split)

    # Clean up temporary directories
    shutil.rmtree(temp_speech_dir)
    shutil.rmtree(temp_non_speech_dir)

    print("Processing complete. Check speech_22050_2 and non_speech_22050_2 directories.")