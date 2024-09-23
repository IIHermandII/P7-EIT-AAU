import numpy as np
import librosa
import soundfile as sf
import scipy.stats
import joblib

# Load the audio file
file_path = 'Data/DiverCut.wav'
audio, sample_rate = librosa.load(file_path, sr=None)

def extract_features(y, sr):  #  `extract_features` is the function defined to extract the features used by the model

    features = {}

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i, mfcc in enumerate(mfccs):
        features[f'MFCC{i+1}'] = np.mean(mfcc)

    # MFCCs variability
    features['MFCC_Var'] = np.var(mfccs, axis=1).mean()  # Variance of MFCCs

    # Other features
    # Contrast Features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['Spectral_Contrast_Mean'] = np.mean(spectral_contrast, axis=1).mean()
    features['Spectral_Contrast_Var'] = np.var(spectral_contrast, axis=1).mean()

    #  Chroma Frequencies (Chroma Features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma, axis=1).mean()
    features['Chroma_Var'] = np.var(chroma, axis=1).mean()

    # Spectral Flatness
    sfm = librosa.feature.spectral_flatness(y=y)
    features['SFM'] = np.mean(sfm)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['Spectral_Centroid'] = np.mean(spectral_centroids)

    # Spectral Spread
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['Spectral_Spread'] = np.mean(spectral_bandwidth)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['ZCR'] = np.mean(zcr)

    # Short-Time Energy
    frame_length = 1024
    hop_length = 256
    ste = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    features['STE'] = np.mean(ste)

    # Spectral Skewness
    stft = np.abs(librosa.stft(y))
    skewness = scipy.stats.skew(stft, axis=0)
    features['Spectral_Skewness'] = np.mean(skewness)

    return features



# Path to where the model file is saved
model = 'finalized_model.sav'  #  `model` is the trained classifier

# Load the model from disk
loaded_model = joblib.load(model)


#  This function reduces the volume of an audio segment by a specified amount in decibels (dB_reduction).
def lower_volume_by_dB(audio_segment, dB_reduction):
    amplitude_ratio = 10 ** (-dB_reduction / 20)
    return audio_segment * amplitude_ratio

def process_and_lower_volume(audio, sample_rate, model, window_length=0.3, hop_length=0.15, dB_reduction=12):
    #  This line calculates how many samples are in each window.
    window_length_samples = int(window_length * sample_rate)
    # This line calculates how many samples the hop length contains.
    hop_length_samples = int(hop_length * sample_rate)
    # Create a copy of the audio to modify
    processed_audio = np.copy(audio)

    for start_sample in range(0, len(audio), hop_length_samples):
        end_sample = min(start_sample + window_length_samples, len(audio))
        segment = audio[start_sample:end_sample]


        features = extract_features(segment, sample_rate)
        feature_array = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(feature_array)[0]  # The model returns the label of the segment

        if prediction == 'breathing':
            # Lower the volume of the segment classified as 'breathing'
            processed_audio[start_sample:end_sample] = lower_volume_by_dB(segment, dB_reduction)

    return processed_audio

lower_audio = (process_and_lower_volume
               (audio, sample_rate, loaded_model, window_length=0.3, hop_length=0.15))


# Save the modified audio to a new file
output_path = 'LoweredBreathingCut.wav'
sf.write(output_path, lower_audio, sample_rate)
