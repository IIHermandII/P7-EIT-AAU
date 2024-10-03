import os
import sys
import wave 
import pylab
import datetime
import matplotlib.pyplot as plt
import numpy as np  
from scipy import signal
from scipy.io import wavfile
import glob

AudioClipLength = 100 # in ms 

def AudioInfo(wav_file, file1):
    print("--------"+ file1 + " INFO-----------")
    sampleFrequensy = wav_file.getframerate()
    numberOfAudioFrames = wav_file.getnframes()
    SterioOrMono = wav_file.getnchannels()
    print("sampleFrequensy: " + str(sampleFrequensy) + "\nnumberOfAudioFrames: " + str(numberOfAudioFrames) + "\nSterioOrMono: " + str(SterioOrMono))

def AudioPlot(wav_file,file1):
    signal = wav_file.readframes(-1)
    signal = np.fromstring(signal, np.int16)
    # If Stereo
    if wav_file.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    sampleFrequensy = wav_file.getframerate()
    sig = np.frombuffer(wav_file.readframes(sampleFrequensy), dtype=np.int16)
    sig = sig[:]
    sig = sig[25000:32000]

    plt.figure(1)
    plot_a = plt.subplot(211)
    plot_a.plot(sig)

    plot_b = plt.subplot(212)
    plot_b.specgram(sig, NFFT=1024, Fs=sampleFrequensy, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')
    sound_info, frame_rate = get_wav_info(wav_file)

def MakeSmaleWavFile(file1):
    from pydub import AudioSegment
    from pydub import AudioSegment
    t1 = 1 * 1000 #Works in milliseconds
    t2 = 3 * 1000
    newAudio = AudioSegment.from_wav(file1)
    newAudio = newAudio[t1:t2]
    newAudio.export('NewData.wav', format="wav") #Exports to a wav file in the current path

# Uses env var 
def FindDataDir():
    envP7Path = os.getenv("P7Path")
    if envP7Path is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7Path' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7Path + "\\Data\\Refined data\\Labeled data"
    os.chdir(workDir) # Changes Dir to working Dir ( Labeled data )
    print(os.getcwd())
    BreathInDir     = workDir + "\\Breath in"
    BreathOutDir    = workDir + "\\Breath out"
    CrossTalkDir    = workDir + "\\Cross talk"
    VoiceDir        = workDir + "\\Voice"
    OtherDir        = workDir + "\\Other"
    DirList = [BreathInDir, BreathOutDir, CrossTalkDir, VoiceDir, OtherDir]
    return DirList

def MakeWorkingDir(soundKlipLength):
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(currentTime)
    DataId = str(soundKlipLength) + "ms_" + str(currentTime)
    workDir = os.getcwd() 

    biDir = workDir + f'\\PROCESSED DATA\\bi\\bi_{DataId}'
    boDir = workDir + f'\\PROCESSED DATA\\bo\\bo_{DataId}'
    ctDir = workDir + f'\\PROCESSED DATA\\ct\\ct_{DataId}'
    voDir = workDir + f'\\PROCESSED DATA\\vo\\vo_{DataId}'

    os.makedirs(biDir)
    os.makedirs(boDir)
    os.makedirs(ctDir)
    os.makedirs(voDir)

def SplitData(AudioClipLength, DirList, ProcessedDirList):
    from pydub import AudioSegment
    FileNameList = ["2_Breath_in.wav", "2_Breath_out.wav", "2_Cross_talk.wav", "2_Voice.wav"]
    for i in range(len(ProcessedDirList)):
        print(os.getcwd())
        AudioFile = DirList[i] + "\\" + FileNameList[i]
        print(AudioFile)
        AudioFile = AudioSegment.from_wav(AudioFile)
        TotalLengthInMs = len(AudioFile)
        print("LENGTH : " + str(TotalLengthInMs))
        t1 = 1
        t2 = AudioClipLength
        working = 1
        while(working):
            if TotalLengthInMs > t2:
                NewAudioFile = AudioFile[t1:t2]
                DestintFile = ProcessedDirList[i] + "\\Data" + str(t1) + "-" + str(t2) + ".wav"
                NewAudioFile.export(DestintFile, format="wav") #Exports to a wav file in the current path
            else:
                working = 0
            t1 = t2
            t2 = t2 + AudioClipLength

def main():
    
    DirList = FindDataDir()
    ProcessedDirList = MakeWorkingDir(DirList, AudioClipLength)
    SplitData(AudioClipLength, DirList, ProcessedDirList)



if __name__ == "__main__":
    main()

