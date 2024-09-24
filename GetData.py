import os
import sys
import wave 
import pylab
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





def main():
    envP7Path = os.getenv("P7Path")
    if envP7Path is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7Path' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7Path + "\\Data"
    RefinedDataDir = workDir + "\\Refined data"
    os.chdir(RefinedDataDir)
    print(os.getcwd())
    file1 = "2.wav"

    wav_file = wave.open(file1,'r')
    AudioInfo(wav_file, file1)
    MakeSmaleWavFile(file1)
    #AudioPlot(wav_file, file1)


if __name__ == "__main__":
    main()

