import os
import sys
import wave 
import pylab
import datetime
import matplotlib.pyplot as plt
import numpy as np  
from scipy import signal
from scipy.io import wavfile

def AudioInfo(wav_file, file1):
    print("--------"+ file1 + " INFO-----------")
    sampleFrequensy = wav_file.getframerate()
    numberOfAudioFrames = wav_file.getnframes()
    SterioOrMono = wav_file.getnchannels()
    print("sampleFrequensy: " + str(sampleFrequensy) + "\nnumberOfAudioFrames: " + str(numberOfAudioFrames) + "\nSterioOrMono: " + str(SterioOrMono))
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
    DirList = [BreathInDir, BreathOutDir, CrossTalkDir, VoiceDir]
    return DirList

def MakeWorkingDir(DirList,soundKlipLength):
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(currentTime)
    DataDirList = DirList
    DataIdentyfire = str(soundKlipLength) + "ms_" + str(currentTime)
    workDir = os.getcwd() 
    biDir = workDir + '\\PROCESSED DATA\\bi\\bi_' + DataIdentyfire 
    boDir = workDir + '\\PROCESSED DATA\\bo\\bo_' + DataIdentyfire
    ctDir = workDir + '\\PROCESSED DATA\\ct\\ct_' + DataIdentyfire
    voDir = workDir + '\\PROCESSED DATA\\vo\\vo_' + DataIdentyfire
    os.makedirs(biDir)
    os.makedirs(boDir)
    os.makedirs(ctDir)
    os.makedirs(voDir)
    ProcessedDirList = [biDir, boDir, ctDir, voDir]
    return ProcessedDirList

def SplitData(AudioClipLength, DirList, ProcessedDirList):
    from pydub import AudioSegment
    FileNameList = ["Breath IN (2,4,5,18).wav", "Breath OUT (2,4,5,18).wav", "Cross talk (2,4,5,18).wav", "Voice (2,4,5,18).wav"]
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
    return DestintFile

def fft(DestintFile):
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft
    from scipy.io import wavfile # get the api
    print("##############################################")
    print(DestintFile)
    fs, data = wavfile.read(DestintFile) # load the data
    print(data)
    a = data.T[0] # this is a two channel soundtrack, I get the first track
    b=[(ele/2**16.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    c = fft(b) # calculate fourier transform (complex numbers list)
    d = len(c)/2  # you only need half of the fft list (real signal symmetry)
    plt.plot(abs(c[:(d-1)]),'r') 
    plt.show()

def main():
    AudioClipLength = 100 # in ms 
    DirList = FindDataDir()
    ProcessedDirList = MakeWorkingDir(DirList, AudioClipLength)
    DestintFile = SplitData(AudioClipLength, DirList, ProcessedDirList)
    #fft(DestintFile)    

if __name__ == "__main__":
    main()
