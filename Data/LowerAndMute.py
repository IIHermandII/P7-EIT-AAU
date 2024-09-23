import numpy as np
import librosa
import soundfile as sf
import scipy.stats
import joblib

#  Load the audio
file_path = 'Data/DiverCut.wav'
audio, sample_rate = librosa.load(file_path, sr=None)

#  Extract features
def extract_features(y, sr):
    features = {}
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    features.update({f'MFCC{i+1}': np.mean(mfcc) for i, mfcc in enumerate(mfccs)})
    features['MFCC_Var'] = np.var(mfccs, axis=1).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['Spectral_Contrast_Mean'] = np.mean(spectral_contrast, axis=1).mean()
    features['Spectral_Contrast_Var'] = np.var(spectral_contrast, axis=1).mean()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma, axis=1).mean()
    features['Chroma_Var'] = np.var(chroma, axis=1).mean()
    sfm = librosa.feature.spectral_flatness(y=y)
    features['SFM'] = np.mean(sfm)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['Spectral_Centroid'] = np.mean(spectral_centroids)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['Spectral_Spread'] = np.mean(spectral_bandwidth)
    zcr = librosa.feature.zero_crossing_rate(y)
    features['ZCR'] = np.mean(zcr)
    frame_length = 1024
    hop_length = 256
    ste = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    features['STE'] = np.mean(ste)
    stft = np.abs(librosa.stft(y))
    skewness = scipy.stats.skew(stft, axis=0)
    features['Spectral_Skewness'] = np.mean(skewness)
    return features

# Load the model from disk
model = 'finalized_model.sav'
loaded_model = joblib.load(model)


#  Equation to lower the volume
def lower_volume_by_dB(audio_segment, dB_reduction):
    amplitude_ratio = 10 ** (-dB_reduction / 20)
    return audio_segment * amplitude_ratio


def process_audio(audio, sample_rate, model, window_length=0.3, hop_length=0.15, dB_reduction=12, mute_reduction=50):
    window_length_samples = int(window_length * sample_rate)
    hop_length_samples = int(hop_length * sample_rate)
    processed_audio = np.copy(audio)
    segment_labels = []

    # First pass: classify each segment and store labels
    for start_sample in range(0, len(audio), hop_length_samples):
        end_sample = min(start_sample + window_length_samples, len(audio))
        segment = audio[start_sample:end_sample]
        features = extract_features(segment, sample_rate)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        segment_labels.append(prediction)

    # Second pass: adjust volume or mute based on distance to voice segments
    for i, label in enumerate(segment_labels):
        start_sample = i * hop_length_samples
        end_sample = min(start_sample + window_length_samples, len(audio))

        if label == 'breathing':
            # Find the nearest voice segment
            distances = [abs(i - j) for j, l in enumerate(segment_labels) if l == 'voice']
            if distances:
                min_distance = min(distances)
                if min_distance <= 4:
                    # Mute if the breathing segment is within 4 segments of any voice segment
                    processed_audio[start_sample:end_sample] = lower_volume_by_dB(audio[start_sample:end_sample], mute_reduction)
                elif min_distance > 4:
                    # Lower volume if the breathing segment is further than 4 segments from all voice segments
                    processed_audio[start_sample:end_sample] = lower_volume_by_dB(audio[start_sample:end_sample], dB_reduction)
            else:
                # Default case if no voice segments are found at all in the audio
                processed_audio[start_sample:end_sample] = lower_volume_by_dB(audio[start_sample:end_sample], dB_reduction)

    return processed_audio



processed_audio = process_audio(audio, sample_rate, loaded_model, window_length=0.3, hop_length=0.15, dB_reduction=12)

# Save the modified audio
output_path = 'LowerandMute.wav'
sf.write(output_path, processed_audio, int(sample_rate))



