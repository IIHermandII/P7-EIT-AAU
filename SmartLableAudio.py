import os
import numpy as np
import librosa
import soundfile as sf
import scipy.stats
import joblib
import pandas as pd
from tqdm import tqdm,trange
import warnings
from termcolor import colored
from scipy import signal
import sys
os.system('color')  # This enables ANSI escape sequences in the terminal.

def GetDataFiles(RootDir,SoundFile):
    filePath = RootDir + "\\SoundFile\\" + SoundFile 
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

def MarkSegment(AudioDataVector, SampleRate, Model, track, WindowLength=0.2, OverLab=0.0): # sr = 1 sec = 44100
    #WL and HL is in sec - then convert to samples
    #############################################
    #       VERSION 1 large Voice               #
    #                                           #
    #############################################
    WindowLengthSampels = int(WindowLength * SampleRate)   # window length * sample rate
    HopLengthSampel = int(WindowLengthSampels * (1-OverLab))   # hop length, OL = over lap

    StartSample = 0
    EndSample = 0
    prediction_list = []
    PermanentStop = 0
    VoiceSergeSteps = 2
    while StartSample < len(AudioDataVector):
        if PermanentStop:
            break
        ConfidenseList = []
        VoiceCounter = 0
        VoiceFount = 0
        CorsstalkFount = 0
        BreathinFount = 0
        BreathoutFount = 0
        MaskFount = 0
        HopLengthMultyplier = 1
        prediction =""
        OldStart = StartSample
        while 1:
            EndSample = min(StartSample + HopLengthSampel, len(AudioDataVector))
            window = AudioDataVector[StartSample:EndSample]
            features_df, features = FeatureExtraction(window, SampleRate, track)
            selectedFeatures = features_df.drop(['Filename','Label'], axis=1)
            confidence = Model.predict_proba(selectedFeatures) # [.1,2.,.4,.6..]
            prediction = Model.predict(selectedFeatures)
            
            # V
            if VoiceFount:          # only when a voice is found 
                HopLengthMultyplier += 1
                if prediction == "V":   # evry time a non voice is found 
                    VoiceCounter = 0   # We count one up  # Important is here  # We reset so now we only exit when no more voice 
                else:
                    VoiceCounter += 1
                if VoiceCounter > VoiceSergeSteps:
                    break
            if prediction == "V":   # We detect what is most likley a vois element
                VoiceFount = 1      # We open the case voise found
            
            # CT
            if VoiceFount != 1:
                if CorsstalkFount:
                    if prediction != "CT":
                        break
                if BreathinFount:
                    if prediction != "BI":
                        break
                if BreathoutFount:
                    if prediction != "BO":
                        break
                if MaskFount:
                    if prediction != "M":
                        break
                if prediction == "CT":
                    CorsstalkFount = 1
                    HopLengthMultyplier += 1
                if prediction == "BI":
                    BreathinFount = 1
                    HopLengthMultyplier += 1
                if prediction == "BO":
                    BreathoutFount = 1
                    HopLengthMultyplier += 1
                if prediction == "M":
                    MaskFount = 1
                    HopLengthMultyplier += 1
                
            if EndSample > 20000000:
                PermanentStop = 1
                break

            ConfidenseList.append(confidence)
            if VoiceFount != 1 and CorsstalkFount != 1 and BreathinFount !=1 and BreathoutFount != 1 and MaskFount != 1: #Yes must be 1 since 2 is also true and would break the logic
                break
            StartSample = EndSample

        # print(StartSample)
        # print(EndSample)

        if VoiceFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel*(VoiceSergeSteps+1))/SampleRate) + "\t" + "V")
            # StartSample = EndSample - HopLengthSampel*(VoiceSergeSteps+1)
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "V")
            StartSample = EndSample
        elif CorsstalkFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "CT")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "CT")
            StartSample = EndSample
        elif BreathinFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "BI")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "BI")
            StartSample = EndSample
        elif BreathoutFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "BO")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "BO")
            StartSample = EndSample
        else:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "M")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "M")
            StartSample = EndSample


        print("BI        BO        CT        M         V")
        for i in ConfidenseList:
            DataList = [i[0][0], i[0][1], i[0][2], i[0][3], i[0][4]]
            SortedDataList = sorted(DataList, reverse=True)
            for val in DataList:
                # Check if value is greater than 0.9 and apply bold and red formatting
                if   val == SortedDataList[0]:
                    print(colored(f"{val:.7f}", 'green', attrs=['bold']), end=" ")
                elif val == SortedDataList[1]:
                    print(colored(f"{val:.7f}", 'light_green'), end=" ")
                elif val == SortedDataList[2]:
                    print(colored(f"{val:.7f}", 'yellow'), end=" ")
                elif val == SortedDataList[3]:
                    print(colored(f"{val:.7f}", 'red'), end=" ")
                else:
                    print(colored(f"{val:.7f}", 'light_red'), end=" ")

            print("\n")
    return prediction_list

def SmartLableSoundFile(AudioDataVector, SampleRate, Model, track, WindowLength=0.2, OverLab=0.0): # sr = 1 sec = 44100
    #WL and HL is in sec - then convert to samples
    #############################################
    #       VERSION 1 large Voice               #
    #                                           #
    #############################################
    WindowLengthSampels = int(WindowLength * SampleRate)   # window length * sample rate
    HopLengthSampel = int(WindowLengthSampels * (1-OverLab))   # hop length, OL = over lap

    StartSample = 0
    LastVoiceSample = 0
    VoiceCounter = 0
    LastEndSample = 0
    EndSample = 0
    prediction_list = []
    PermanentStop = 0
    VoiceSergeSteps = 2
    while StartSample < len(AudioDataVector):
        if PermanentStop:
            break
        ConfidenseList = []
        VoiceFount = 0
        CorsstalkFount = 0
        BreathinFount = 0
        BreathoutFount = 0
        MaskFount = 0
        HopLengthMultyplier = 1
        prediction =""
        OldStart = StartSample
        while 1:
            LastEndSample  = EndSample
            EndSample = min(StartSample + HopLengthSampel, len(AudioDataVector))
            window = AudioDataVector[StartSample:EndSample]
            features_df, features = FeatureExtraction(window, SampleRate, track)
            selectedFeatures = features_df.drop(['Filename','Label'], axis=1)
            confidence = Model.predict_proba(selectedFeatures) # [.1,2.,.4,.6..]
            prediction = Model.predict(selectedFeatures)
            
            
            # CT
            if VoiceFount:
                if prediction != "V":
                    VoiceCounter += 1
                    HopLengthMultyplier += 1
                else:
                    LastVoiceSample = EndSample
                    VoiceCounter = 0
                    HopLengthMultyplier += 1
            else:
                if CorsstalkFount:
                    if prediction != "CT":
                        EndSample = LastEndSample
                        break
                if BreathinFount:
                    if prediction != "BI":
                        EndSample = LastEndSample
                        break
                if BreathoutFount:
                    if prediction != "BO":
                        EndSample = LastEndSample
                        break
                if MaskFount:
                    if prediction != "M":
                        EndSample = LastEndSample
                        break
                if prediction == "V":
                    VoiceFount = 1
                    HopLengthMultyplier += 1
                    LastVoiceSample = EndSample
                if prediction == "CT":
                    CorsstalkFount = 1
                    HopLengthMultyplier += 1
                if prediction == "BI":
                    BreathinFount = 1
                    HopLengthMultyplier += 1
                if prediction == "BO":
                    BreathoutFount = 1
                    HopLengthMultyplier += 1
                if prediction == "M":
                    MaskFount = 1
                    HopLengthMultyplier += 1
                
            if EndSample > 20000000:
                PermanentStop = 1
                break
            
            if VoiceCounter > 3:
                EndSample = LastVoiceSample
                LastVoiceSample = 0
                VoiceCounter = 0
                break



            ConfidenseList.append(confidence)
            if VoiceFount != 1 and CorsstalkFount != 1 and BreathinFount !=1 and BreathoutFount != 1 and MaskFount != 1 and VoiceCounter == 0: #Yes must be 1 since 2 is also true and would break the logic
                break
            StartSample = EndSample

        # print(StartSample)
        # print(EndSample)

        if VoiceFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel*(VoiceSergeSteps+1))/SampleRate) + "\t" + "V")
            # StartSample = EndSample - HopLengthSampel*(VoiceSergeSteps+1)
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "V")
            StartSample = EndSample
        elif CorsstalkFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "CT")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "CT")
            StartSample = EndSample
        elif BreathinFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "BI")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "BI")
            StartSample = EndSample
        elif BreathoutFount:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "BO")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "BO")
            StartSample = EndSample
        else:
            # prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample-HopLengthSampel)/SampleRate) + "\t" + "M")
            # StartSample = EndSample - HopLengthSampel
            prediction_list.append(str(OldStart/SampleRate) + "\t" + str((EndSample)/SampleRate) + "\t" + "M")
            StartSample = EndSample


        print("BI        BO        CT        M         V")
        for i in ConfidenseList:
            DataList = [i[0][0], i[0][1], i[0][2], i[0][3], i[0][4]]
            SortedDataList = sorted(DataList, reverse=True)
            for val in DataList:
                # Check if value is greater than 0.9 and apply bold and red formatting
                if   val == SortedDataList[0]:
                    print(colored(f"{val:.7f}", 'green', attrs=['bold']), end=" ")
                elif val == SortedDataList[1]:
                    print(colored(f"{val:.7f}", 'light_green'), end=" ")
                elif val == SortedDataList[2]:
                    print(colored(f"{val:.7f}", 'yellow'), end=" ")
                elif val == SortedDataList[3]:
                    print(colored(f"{val:.7f}", 'red'), end=" ")
                else:
                    print(colored(f"{val:.7f}", 'light_red'), end=" ")

            print("\n")
    return prediction_list

def RawLablesoundFile(y, sr, model, track, WL=0.2, OL=0): # sr = 1 sec = 44100
    #WL and HL is in sec - then convert to samples
    WL_samples = int(WL * sr)   # window length * sample rate
        # hop length, OL = over lap
    HL_samples = int(WL_samples * (1-OL))
    prediction_list = []
    conf = []
    #Loop through all start samples of windows in the audio track
    for start_sample in tqdm(range(0, len(y), HL_samples)):
        #Find end sample from start sample using WL (handle edge case when track ends)
        end_sample = min(start_sample + WL_samples, len(y))
        #Select samples
        window = y[start_sample:end_sample]
        #Extract all features
        features_df, features = FeatureExtraction(window, sr, track)
        #Select the features the model is trained on
        selectedFeatures = features_df.drop(['Filename','Label'], axis=1)
        #Run prediction on the selected samples
        prediction = model.predict(selectedFeatures)
        prediction_list.append(str(start_sample/sr) + "\t" + str(end_sample/sr) + "\t" + str(prediction[0]))

    return prediction_list

def MakeSoundFileFromLableFile(RootDir, filePath, NewName):
    LableFileLOcation = RootDir + filePath
    print("Working dir: ")
    print(LableFileLOcation)
    y, sr = GetDataFiles(RootDir,"SoundFile.wav")
    print(len(y))
    print(sr)
    print(len(y)/(sr*60))
    muted_audio = np.copy(y)
    end_gain = 0
    f = open(LableFileLOcation, "r")
    for i, lines in enumerate(f):
        Data = lines.split()
        #print(Data[2])
        if Data[2] in ['BI','BO','M','CT']:  
            #print("Segment mute")
            #muted_audio[int(float(Data[0])*44100):int(float(Data[1])*44100)] = GainFactor*muted_audio[int(float(Data[0])*44100):int(float(Data[1])*44100)]  # Mute this segment
            start_idx = int(float(Data[0]) * 44100)
            end_idx = int(float(Data[1]) * 44100)
            muted_audio[start_idx:end_idx] = muted_audio[start_idx:end_idx] * 0.0
        else:
            start_idx = int((float(Data[0]) * 44100) - 8820)
            end_idx = int((float(Data[1]) * 44100) + 8820)
            #window = np.hanning(end_idx - start_idx)
            window = signal.windows.tukey(end_idx - start_idx,alpha=0.25)
            muted_audio[start_idx:end_idx] = muted_audio[start_idx:end_idx] * window

    sf.write(RootDir + "\\Outputs\\" + NewName, muted_audio, sr)

def main():
    warnings.filterwarnings("ignore")
    RootDir = sys.argv[1]
    y, sr = GetDataFiles(RootDir,"SoundFile.wav")
    print(y)
    pipe = joblib.load(RootDir + "\\Models\\SVM_model_trainset.sav")
    print("Model loaded")   

    print("Making Smart lable file ") 
    SmartPredictionList = SmartLableSoundFile(y,sr,pipe,"2")
    f = open(RootDir + "\\Outputs\\Predictions_2_Smart(SVM).txt", "w")
    for pred in SmartPredictionList:
        f.write(pred + "\n")
    f.close()

    print("Making Raw lable file ")
    OrignPredictionList = RawLablesoundFile(y,sr,pipe,"2")
    f = open(RootDir + "\\Outputs\\Predictions_2_Raw(SVM).txt", "w")
    for pred in OrignPredictionList:
        f.write(pred + "\n")
    f.close()

    print("Making Soundfiles...")
    MakeSoundFileFromLableFile(RootDir , "\\Outputs\\Predictions_2_Smart(SVM).txt","smart.wav")
    MakeSoundFileFromLableFile(RootDir , "\\Outputs\\Predictions_2_Raw(SVM).txt","raw.wav")

if __name__ == "__main__":
    main()