import os
import numpy as np
import librosa
import soundfile as sf
import scipy.stats
import joblib
import pandas as pd
from tqdm import tqdm,trange

import warnings

def GetDataFiles(fileNo):
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    filePath = envP7RootDir + "\\Data\\Refined data\\" + fileNo 
    y, sr = librosa.load(filePath, sr=None)

    print("Loaded file: ", filePath)

    return y, sr

def FeatureExtraction(y, sr, track):
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

    features['Filename'] = track
    features['Label'] = '_'
    
    features_df = pd.DataFrame([features])
    column_order = ['Filename', 'Label'] + [f'LPC{i + 2}' for i in range(5)] +  [f'MFCC{i + 1}' for i in range(13)] + \
                ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var', 'SFM', 'Spectral_Spread', 'Spectral_Skewness', 'Spectral_Centroid', 'Chroma_Mean', 'Chroma_Var', 'ZCR', 'STE', 'RMS']
    features_df = features_df[column_order]
    
    return features_df, features

def mute_segments(y, sr, model, track, WL=0.2, OL=0.75):
    #WL and HL is in sec - then convert to samples
    WL_samples = int(WL * sr)
    HL_samples = int(WL_samples * (1-OL))

    # Classify each segment and mute if labeled as 'breathing'
    muted_audio = np.copy(y)

    prediction_list = []
    all_features = []
    #Loop through all start samples of windows in the audio track
    for start_sample in tqdm(range(0, len(y), HL_samples)):
        #Find end sample from start sample using WL (handle edge case when track ends)
        end_sample = min(start_sample + WL_samples, len(y))
        #Select samples
        window = y[start_sample:end_sample]
        #Extract all features
        features_df, features = FeatureExtraction(window, sr, track)
        #Select the features the model is trained on
        selectedFeatures = features_df[model[1].get_feature_names_out()]
        #Run prediction on the selected samples
        prediction = model.predict(selectedFeatures)
        #features['Label'] = prediction[0]

        #all_features.append(features)
        prediction_list.append(str(prediction + "| Start:" + str(start_sample) + " Stop:" + str(end_sample)))

        #Remove audio if the audio is not voice
        if prediction[0] == 'BI' or prediction[0] == 'BO' or prediction[0] == 'M':
            muted_audio[start_sample:end_sample] = 0.25*muted_audio[start_sample:end_sample]  # Mute this segment

    # features_df = pd.DataFrame(all_features)
    # column_order = ['Filename', 'Label'] + [f'LPC{i + 2}' for i in range(5)] +  [f'MFCC{i + 1}' for i in range(13)] + \
    #             ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var', 'SFM', 'Spectral_Spread', 'Spectral_Skewness', 'Spectral_Centroid', 'Chroma_Mean', 'Chroma_Var', 'ZCR', 'STE', 'RMS']
    # features_df = features_df[column_order]

    return muted_audio, prediction_list, features_df

def main():
    warnings.filterwarnings("ignore")
    envP7RootDir = os.getenv("P7RootDir")
    #Type name of track to be processed
    track = "4.wav"

    #Uncomment for loop and indent from "Indent start" to "Indent end" for data file generation
    #for i in range(9,23):
    #Indent start - 
    #    track = str(i)
    y, sr = GetDataFiles(track)

    # Load the model from disk
    pipe = joblib.load('Models\\LR_model_trainset.sav')

    print("Model loaded")

    #Process audio - 12dB attenuation to BI, BO, M segments
    newAudio, predictions, all_features = mute_segments(y,sr,pipe,track)

    print("Audio processed")
   
    #CSVname = envP7RootDir + "\\Data\\CSV files self\\" + track + ".csv"
    #all_features.to_csv(CSVname, index=False)

    #Indent end

    #print("Model self labelled data output")

    #Export predictions document (Pred | start samp | end samp)
    f = open("Predictions.txt", "w")
    for pred in predictions:
        f.write(pred + "\n")
    f.close()

    print("Predictions data output")

    # Save the modified audio to a new file
    output_path = envP7RootDir + "\\Data\\Processed audio\\" + track + " (processed).wav"
    sf.write(output_path, newAudio, sr)

    print("Processed audio output \nSCRIPT DONE")

if __name__ == "__main__":
    main()