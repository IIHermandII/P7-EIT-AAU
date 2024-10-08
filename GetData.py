import os
import sys
import wave 
import pylab
import datetime
import matplotlib.pyplot as plt
import numpy as np  
from scipy import signal
from scipy.io import wavfile

def FindDataDir():
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7RootDir + "\\Data\\Refined data\\Labeled data"
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
    envP7LocalData = os.getenv("P7LocalData")
    if envP7LocalData is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7LocalData' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    os.chdir(envP7LocalData)
    workDir = os.getcwd()
    if not os.path.exists("bi"):
        os.mkdir("bi")
        os.mkdir("bo")
        os.mkdir("ct")
        os.mkdir("vo")
    biDir = workDir + '\\bi\\bi_' + DataIdentyfire 
    boDir = workDir + '\\bo\\bo_' + DataIdentyfire
    ctDir = workDir + '\\ct\\ct_' + DataIdentyfire
    voDir = workDir + '\\vo\\vo_' + DataIdentyfire
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

def main():
    AudioClipLength = 100 # in ms 
    DirList = FindDataDir()
    ProcessedDirList = MakeWorkingDir(DirList, AudioClipLength)
    DestintFile = SplitData(AudioClipLength, DirList, ProcessedDirList)  

if __name__ == "__main__":
    main()

