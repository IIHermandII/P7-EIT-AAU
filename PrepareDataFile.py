import os
import re
import librosa
import numpy as np
import pandas as pd
import scipy.stats
import re   # Import the regular expression module

def FindPreparedDataDir():
    envP7Path = os.getenv("P7Path")
    if envP7Path is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7Path' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7Path + "\\Data\\Refined data\\Labeled data\\PROCESSED DATA"
    os.chdir(workDir) # Changes Dir to working Dir ( Labeled data )
    print(os.getcwd())
    BreathInDir     = workDir + "\\bi"
    BreathOutDir    = workDir + "\\bo"
    CrossTalkDir    = workDir + "\\ct"
    VoiceDir        = workDir + "\\vo"
    DirList = [BreathInDir, BreathOutDir, CrossTalkDir, VoiceDir]
    for dir in DirList:
        dirDats = []
        i=0
        for subDir in os.listdir(dir):
            onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', subDir)
            new_string = str(onlyDate).replace("-", "")
            new_string = new_string.replace("_","")
            new_string = new_string.strip("[]'")
            dirDats.append([int(new_string),i])
            i = i + 1
    
    print(sorted(dirDats,key=lambda l:l[1],reverse=True)) 
    print("---------------------------")      
    print(dirDats)  
    #print(dirDats)
    return DirList


def FeatureExtraction():
    print("Main")
    dummy = "Data1-100.wav"
    print("WORK DIR: " + os.getcwd())
    y, sr = librosa.load(dummy, sr=None) # sr=none preserve native sampling rate
    features = {}

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    for i, mfcc in enumerate(mfccs):
        features[f'MFCC{i+1}'] = np.mean(mfcc)
    # MFCCs variability
    features['MFCC_Var'] = np.var(mfccs, axis=1).mean()  # Variance of MFCCs

    # Contrast Features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['Spectral_Contrast_Mean'] = np.mean(spectral_contrast, axis=1).mean()
    features['Spectral_Contrast_Var'] = np.var(spectral_contrast, axis=1).mean()

    #  Chroma Frequencies
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

def main():
    FindPreparedDataDir()
    # features = FeatureExtraction()
    # all_features = []
    # features = FeatureExtraction()
    # features['Filename'] = "File Name"
    # features['Label'] = "Lable"
    # all_features.append(features)
    # features_df = pd.DataFrame(all_features)

    # column_order = ['Filename', 'Label'] + [f'MFCC{i + 1}' for i in range(5)] + \
    #             ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var',  'Chroma_Mean', 'Chroma_Var', 'STE', 'Spectral_Skewness']

    # features_df = features_df[column_order]

    # features_df.to_csv('audio_features_with_labels.csv', index=False)
    # print('Feature extraction completed and saved to audio_features_with_labels.csv.')

if __name__ == "__main__":
    main()
