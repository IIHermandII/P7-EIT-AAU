import librosa
import soundfile as sf

# Load the audio file
audio_file = "Data/23.wav"
audio_data, sr = librosa.load(audio_file, sr=None)

# Segment the audio into 1-second intervals
segment_duration = sr  # each 1-second segment will contain exactly sr samples.
segments = [audio_data[i:i + segment_duration] for i in range(0, len(audio_data), segment_duration)]

# Save segmented audio files
for i, segment in enumerate(segments):
    sf.write(f"segment_{i}.wav", segment, sr)
