import os
from pydub import AudioSegment


def MakeWorkingDir():
    workingdir = os.getenv("P7Path") + '\\Data\\Refined data'
    ProcessedDirList = []
    for file in os.listdir(workingdir):
        if '(done)' in file:
            ProcessedDirList.append(file)
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


def SplitData(AudioClipLength, DirList, ProcessedDirList):


    FileNameList = os.listdir(DirList[0])  # Assuming all files are in the first directory of DirList
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
    Unlabeled_list = MakeWorkingDir()
    

if __name__ == "__main__":
    AudioClipLength = 100
    main()