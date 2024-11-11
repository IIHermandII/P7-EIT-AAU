import os, re
import sys
from ConvertToZip import convertToZip
import wave 
import pylab
import datetime
import matplotlib.pyplot as plt
import numpy as np  
from scipy import signal
from scipy.io import wavfile
import glob
import zipfile
from pydub import AudioSegment


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
    plt.show()

def MakeSmaleWavFile(file1):
    from pydub import AudioSegment
    from pydub import AudioSegment
    t1 = 1 * 1000 #Works in milliseconds
    t2 = 3 * 1000
    newAudio = AudioSegment.from_wav(file1)
    newAudio = newAudio[t1:t2]
    newAudio.export('NewData.wav', format="wav") #Exports to a wav file in the current path

def FindDataDir(): # Uses env var 
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

def SplitData(AudioClipLength, DirList, ProcessedDirList, zip_path=None):


    def extract_zip(zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

    # If zip_path is provided, extract it
    if zip_path:
        extract_to = 'temp_extracted_files'
        extract_zip(zip_path, extract_to)
        DirList = [os.path.join(extract_to, d) for d in DirList]
        ProcessedDirList = [os.path.join(extract_to, d) for d in ProcessedDirList]

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
                dest_dir = os.path.dirname(DestintFile)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                NewAudioFile.export(DestintFile, format="wav") # Exports to a wav file in the current path
            else:
                working = 0
            t1 = t2
            t2 = t2 + AudioClipLength

def ChooseFile():
    paths = ["bi", "bo", "ct", "vo"]
    directory = os.getenv("P7Path") + "\\Data\\Refined data\\Labeled data\\PROCESSED DATA\\"
    id = input("Do you want to use the newest file or choose a file? (new/choose): ")
    if id == "new":
        MakeWorkingDir(AudioClipLength)
        latest_files = []
        for path in paths:
            full_path = os.path.join(directory, path)
            files = glob.glob(full_path + "\\*")
            latest_time = datetime.datetime.min
            for file in files:
                file_time = os.path.getctime(file)
                file_time = datetime.datetime.fromtimestamp(file_time)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file
            if latest_file:
                latest_files.append(latest_file)
                print(f"\nChosen file path for {path}: {latest_file}")
        if latest_files:
            ProcessedDirList = latest_files
            print("\nProcessed list = " + str(ProcessedDirList))
            return ProcessedDirList
        else:
            print("No files found.")
            return []
    
    else:
        for path in paths:
            full_path = os.path.join(directory, path)
            files = glob.glob(full_path + "\\*")
            for file in files:
                print(f"File: {file}")
        input_path = input("Enter the path of the file you want to use: ")
        ProcessedDirList = []
        for path in paths:
            full_path = os.path.join(directory, path)
            files = glob.glob(full_path + "\\*")
            for file in files:
                if input_path in file:
                    print(f"Chosen file path: {file}")
                    ProcessedDirList.append(file)
                    break
        return ProcessedDirList
    
def GetNewestDataFileNamer(x):
    #Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    if x == 'labeled':
        #Enter CSV directory, change the directory for labeled data and unlabeled data
        workDir = envP7RootDir + "\\Data\\CSV files"
    else:
        workDir = envP7RootDir + "\\Data\\Refined data\\Unlabeled data\\PROCESSED DATA"
    
    #Find all dates from the files
    dirDates = []
    for file in os.listdir(workDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "")
        new_string = new_string.replace("_","")
        new_string = new_string.strip("[]'")
        dirDates.append([int(new_string),file])
    
    #Sort dates and return newest
    dirDates = sorted(dirDates,key=lambda l:l[1],reverse=True) # Take oldest data first i belive 
    return(workDir + "\\" + dirDates[0][1])

def main():
    
    DirList = FindDataDir()
    ProcessedDirList = ChooseFile()
    SplitData(AudioClipLength, DirList, ProcessedDirList)
    for processed_dir in ProcessedDirList:
        print(processed_dir)
        convertToZip(processed_dir)
    print(ProcessedDirList)
    

if __name__ == "__main__":
    main()


