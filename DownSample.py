import os
import librosa
import soundfile as sf

def downsample_audio(input_path, output_directory, target_sr_list=[8000, 16000, 24000]):
    y, sr = librosa.load(input_path, sr=None)
    for target_sr in target_sr_list:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        target_directory = os.path.join(output_directory, f"{target_sr}Hz")
        os.makedirs(target_directory, exist_ok=True)
        output_filename = os.path.join(target_directory, f"{os.path.splitext(os.path.basename(input_path))[0]}_{target_sr}Hz.wav")
        sf.write(output_filename, y_resampled, target_sr)

# Directory containing the original audio files
input_directory = 'C:/Users/caspe/Aalborg Universitet/Marcus Mogensen - 7. semester/P7/Data/Refined data'

# Directory to save the downsampled audio files
output_directory = 'C:/Users/caspe/Aalborg Universitet/Marcus Mogensen - 7. semester/P7/Data/Downsampled data'

# Downsample each audio file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        input_path = os.path.join(input_directory, filename)
        downsample_audio(input_path, output_directory)
