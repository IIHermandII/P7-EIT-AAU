import os
import librosa
import soundfile as sf
import tqdm

def main():
    envP7RootDir = os.getenv("P7RootDir")

    dirPath = envP7RootDir + "\\Data\\Label clips"

    newSR = 500

    for root, _, files in tqdm.tqdm(os.walk(dirPath, topdown=True)):
        for file in files:
            y, sr = librosa.load(root+"\\"+file)
            y_hat = librosa.resample(y,orig_sr=sr,target_sr=newSR)
            newRoot = root.replace("Label clips","Label clips 500Hz")
            sf.write(newRoot+"\\"+file, y_hat, newSR) 

if __name__ == "__main__":
    main()