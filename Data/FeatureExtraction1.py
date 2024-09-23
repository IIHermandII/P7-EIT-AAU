import os
import librosa
import numpy as np
import pandas as pd
import scipy.stats

def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
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

    # Chroma Frequencies (Chroma Features)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma, axis=1).mean()
    features['Chroma_Var'] = np.var(chroma, axis=1).mean()

    # Spectral Flatness
    # sfm = librosa.feature.spectral_flatness(y=y)
    # features['SFM'] = np.mean(sfm)

    # Spectral Centroid
    # spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    # features['Spectral_Centroid'] = np.mean(spectral_centroids)

    # Spectral Spread
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # features['Spectral_Spread'] = np.mean(spectral_bandwidth)

    # Zero-Crossing Rate
    # zcr = librosa.feature.zero_crossing_rate(y)
    # features['ZCR'] = np.mean(zcr)

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

# Directory containing the audio files
directory_path = 'FirstTraining'

all_features = []  # List to store all features for each file
for filename in os.listdir(directory_path):
    if filename.endswith(".wav"):  # Only process WAV files
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {filename}...")
        features = extract_features(file_path)
        features['Filename'] = filename  # Add filename to features at the end
        all_features.append(features)

#  Convert the list of features dictionaries to a DataFrame
features_df = pd.DataFrame(all_features)

#  Specify column order starting with Filename, followed by MFCCs, then other features
column_order = ['Filename'] + [f'MFCC{i+1}' for i in range(5)] + \
               ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var', 'Chroma_Mean', 'Chroma_Var'
                   , 'STE', 'Spectral_Skewness']

#  Reorder DataFrame columns
features_df = features_df[column_order]

#  Save the DataFrame to a CSV file
features_df.to_csv('audio_features.csv', index=False)
print('Feature extraction completed and saved to audio_features.csv.')
