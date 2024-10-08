import os
from os.path import isfile, join
import re
import librosa
import numpy as np
import pandas as pd
import scipy.stats
import re   # Import the regular expression module
import datetime

# Finds newest data works, just use it
def FindPreparedDataDir(workDir):
    os.chdir(workDir) # Changes Dir to working Dir ( Labeled data )
    print(os.getcwd())
    BreathInDir     = workDir + "\\bi"
    BreathOutDir    = workDir + "\\bo"
    CrossTalkDir    = workDir + "\\ct"
    VoiceDir        = workDir + "\\vo"
    DirList = [BreathInDir, BreathOutDir, CrossTalkDir, VoiceDir]
    DeciredFolders = []
    FileName = []
    Category = []
    for dir in DirList:
        dirDats = []
        i=0
        for subDir in os.listdir(dir):
            onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', subDir)
            new_string = str(onlyDate).replace("-", "")
            new_string = new_string.replace("_","")
            new_string = new_string.strip("[]'")
            dirDats.append([int(new_string),i,subDir])
            i = i + 1
        dirDats = sorted(dirDats,key=lambda l:l[1],reverse=True) # Take oldest data first i belive 
        FileName.append(dirDats[0][2])
        DeciredFolders.append("\\" + FileName[-1])
        macth = (re.match(r'^([a-z]{2})_', (FileName[-1])))
        Category.append(macth.group(1))
        
    FullPathToFileList = [[x + y for x, y in zip(DirList, DeciredFolders)],FileName,Category]
    print("\n--------------------DATA COLLECTED FOR USE IN CSV:-----------------------")
    print("\n--------------------DATA DIR LIST----------------:-----------------------")
    print(FullPathToFileList[0]) 
    print("\n--------------------DATA FILE NAME---------------:-----------------------")
    print(FullPathToFileList[1]) 
    print("\n--------------------DATA CATAGORY----------------:-----------------------")
    print(FullPathToFileList[2]) 
    print("\n-------------------------------------------------------------------------")
    return FullPathToFileList

def FeatureExtraction(WovFile):
    y, sr = librosa.load(WovFile, sr=None) # sr=none preserve native sampling rate
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
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7RootDir + "\\Data\\Refined data\\Labeled data\\PROCESSED DATA"

    all_features = []
    DataDirList = FindPreparedDataDir(workDir) #returns matrix containing all info 
    print("\n\n\nWORKING ON CSV THIS MAY TAKE A FEW ME-NUTS >:D ...")
    for i in range(len(DataDirList[0][:])):
        for WovFile in os.listdir(DataDirList[0][i]): # the folder path
             
            # MAKE THE CVS FILE WITH DATA FROM ALL THE FILES
            FullFilePath = DataDirList[0][i] + "\\" + WovFile
            #print(FullFilePath)
            features = FeatureExtraction(FullFilePath)
            features['Filename'] = DataDirList[1][i]
            features['Label'] = DataDirList[2][i]
            all_features.append(features)

    #COMBINE CSV FILE 
    features_df = pd.DataFrame(all_features)
    column_order = ['Filename', 'Label'] + [f'MFCC{i + 1}' for i in range(5)] + \
                ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var',  'Chroma_Mean', 'Chroma_Var', 'STE', 'Spectral_Skewness']
    features_df = features_df[column_order]
    if not os.path.exists("CSV DATA"):
        os.mkdir("CSV DATA")
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    CSVFileName = 'CSV DATA\\' +currentTime+'_audio_features_with_labels.csv'
    features_df.to_csv(CSVFileName, index=False)
    
    print('CSV file created and saved to:')
    print(workDir + "\\CSV DATA")
    print('CSV saved under the name: ' + currentTime + '_audio_features_with_labels.csv' )




if __name__ == "__main__":
    main()
