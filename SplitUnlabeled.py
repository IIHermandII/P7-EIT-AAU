import os
from pydub import AudioSegment
import librosa


def MakeWorkingDir():
    workingdir = os.getenv("P7Path") + '\\Data\\Refined data'
    ProcessedDirList = []
    for file in os.listdir(workingdir):
        if '(done)' in file:
            ProcessedDirList.append(file.split('(done)')[0].replace(' ', ''))
    print('ProcessedDirList = ')
    print(ProcessedDirList)
    workingdir = workingdir + '\\Unlabeled data'
    for file in ProcessedDirList:
        new_dir = os.path.join(workingdir, file.split('(done)')[0])
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created directory: {new_dir}")
    return ProcessedDirList
        
def ChooseDir(): # Uses env var 
    envP7Path = os.getenv("P7Path")
    if envP7Path is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7Path' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7Path + "\\Data\\Refined data\\Unlabeled data"
    os.chdir(workDir) # Changes Dir to working Dir ( Unlabeled data )
    print(os.getcwd())


def SplitData(AudioClipLength, ProcessedDirList):
    # for audionumber in ProcessedDirList:
    audio_file_path = os.path.join(os.getenv("P7Path"), 'Data', 'Refined data', "16.wav")
    dir_path = os.path.join(os.getenv("P7Path"), 'Data', 'Refined data', 'Unlabeled data', '16')

    # audio_file_path = os.path.join(os.getenv("P7Path"), 'Data', 'Refined data', f"{audionumber}.wav")
    # dir_path = os.path.join(os.getenv("P7Path"), 'Data', 'Refined data', 'Unlabeled data', audionumber)
    
    if not os.path.exists(audio_file_path):
        print(f"File not found: {audio_file_path}")
        # continue

    AudioFile = AudioSegment.from_wav(audio_file_path)
    TotalLengthInMs = len(AudioFile)
    t1 = 0
    t2 = AudioClipLength

    while t1 < TotalLengthInMs:
        if t2 > TotalLengthInMs:
            t2 = TotalLengthInMs
        NewAudioFile = AudioFile[t1:t2]
        DestintFile = os.path.join(dir_path, f"Data_{t1}-{t2}.wav")
        NewAudioFile.export(DestintFile, format="wav")
        print(f"Exported: {DestintFile}")
        t1 = t2
        t2 += AudioClipLength

def main():
    Unlabeled_list = MakeWorkingDir()
    SplitData(AudioClipLength, Unlabeled_list)
    

if __name__ == "__main__":
    AudioClipLength = 100
    main()