import os
import re
import librosa
import numpy as np
import pandas as pd
import datetime
import scipy.stats
from tqdm import trange

def OprationExplanation():
    print("\n\n\n-------------- Labelled data to CSV file --------------")
    print("This file expects the following to work:")
    print("1.\tenv ( Environment variables / System variables):")
    print("\tP7RootDir : C:\\path\\to\\P7  i.g C:\\Users\\emill\\OneDrive - Aalborg Universitet\\P7 ")
    print("2.\tAll files at: P7\\Data\\Label clips. to have same structure")
    print("\t<name>-<number>-...-<number>.wov")

def GetDataFiles():
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    workDir = envP7RootDir + "\\Data\\Label clips"
    DataList = []
    ListOfLabels = []

    for root, _, files in os.walk(workDir, topdown=True):
        for file in files:
            regExExpression = re.match("([A-z])+",file)
            LastDir = os.path.basename(os.path.normpath((root))) + "\\" + file 
            FilePath = os.path.join(root,file)
            DataList.append([FilePath,LastDir,regExExpression[0]])
            if regExExpression[0] not in ListOfLabels:
                ListOfLabels.append(regExExpression[0])
    return DataList, envP7RootDir

def FeatureExtraction(WovFile):
    y, sr = librosa.load(WovFile, sr=None) # sr=none preserve native sampling rate
    features = {}

    ##Frequency
    lpcs = librosa.lpc(y=y,order=6)
    for i, lpc in enumerate(lpcs):
        features[f'LPC{i+1}'] = lpc

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # 5
    for i, mfcc in enumerate(mfccs):
        features[f'MFCC{i+1}'] = np.mean(mfcc)
    # MFCCs variability
    features['MFCC_Var'] = np.var(mfccs, axis=1).mean()  # Variance of MFCCs

    # Contrast Features
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['Spectral_Contrast_Mean'] = np.mean(spectral_contrast, axis=1).mean()
    features['Spectral_Contrast_Var'] = np.var(spectral_contrast, axis=1).mean()

    # Spectral Flatness
    sfm = librosa.feature.spectral_flatness(y=y)
    features['SFM'] = np.mean(sfm)

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['Spectral_Centroid'] = np.mean(spectral_centroids)

    # Spectral Spread
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['Spectral_Spread'] = np.mean(spectral_bandwidth)

    # Spectral Skewness
    stft = np.abs(librosa.stft(y))
    skewness = scipy.stats.skew(stft, axis=0)
    features['Spectral_Skewness'] = np.mean(skewness)

    # Chroma Frequencies
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['Chroma_Mean'] = np.mean(chroma, axis=1).mean()
    features['Chroma_Var'] = np.var(chroma, axis=1).mean()

    ##Time
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['ZCR'] = np.mean(zcr)

    # Short-Time Energy
    frame_length = 1024
    hop_length = 256
    ste = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    features['STE'] = np.mean(ste)

    ##Energy
    # Root-mean-square
    rms = librosa.feature.rms(y=y)
    features['RMS'] = np.mean(rms)

    return features

def MakeCSV(DataList, envP7RootDir):
    print("\n\n\nWORKING ON CSV THIS MAY TAKE A FEW ME-NUTS >:D ...")
    all_features = []
    for i in trange(len(DataList)):
        #print(DataList[i][0])
        FullFilePath = DataList[i][0]
        FileName = DataList[i][1]
        LabelName = DataList[i][2]
        features = FeatureExtraction(FullFilePath)
        features['Filename'] = FileName
        features['Label'] = LabelName
        all_features.append(features)
        #COMBINE CSV FILE 
    features_df = pd.DataFrame(all_features)
    column_order = ['Filename', 'Label'] + [f'LPC{i + 2}' for i in range(5)] +  [f'MFCC{i + 1}' for i in range(13)] + \
                ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var', 'SFM', 'Spectral_Spread', 'Spectral_Skewness', 'Spectral_Centroid', 'Chroma_Mean', 'Chroma_Var', 'ZCR', 'STE', 'RMS']
    features_df = features_df[column_order]
    workDir = envP7RootDir + "\\Data\\CSV files"
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CSVFileName = workDir + '\\' +currentTime+'_audio_features_with_labels.csv'
    features_df.to_csv(CSVFileName, index=False)
    
    print('CSV file created and saved to:')
    print(workDir)
    print('CSV saved under the name: ' + currentTime + '_audio_features_with_labels.csv' )

def main():
    OprationExplanation()
    DataList, envP7RootDir = GetDataFiles()
    MakeCSV(DataList, envP7RootDir)

if __name__ == "__main__":
    main()