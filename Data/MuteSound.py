import numpy as np
import librosa
import soundfile as sf
import scipy.stats
import joblib

# Load the audio file
file_path = 'Data/DiverCut.wav'
audio, sample_rate = librosa.load(file_path, sr=None)

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

def mute_breathing_segments(audio, sample_rate, model, window_length=0.3, hop_length=0.15):
    window_length_samples = int(window_length * sample_rate)
    hop_length_samples = int(hop_length * sample_rate)

    # Classify each segment and mute if labeled as 'breathing'
    muted_audio = np.copy(audio)
    for start_sample in range(0, len(audio), hop_length_samples):
        end_sample = min(start_sample + window_length_samples, len(audio))
        window = audio[start_sample:end_sample]
        features = extract_features(window, sample_rate)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(feature_array)
        if prediction[0] == 'breathing':
            muted_audio[start_sample:end_sample] = 0  # Mute this segment

    return muted_audio

# Process the audio file with the new function
processed_audio = mute_breathing_segments(audio, sample_rate, loaded_model, window_length=0.3, hop_length=0.15)

# Save the modified audio to a new file
output_path = 'Muted.wav'
sf.write(output_path, processed_audio, sample_rate)



