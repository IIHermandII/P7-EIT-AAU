import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extractFeatures(audio):
    """
    This function extracts the features:
    MFCC
    From the given audio file
    """
    y,sr = librosa.load(audio)

    #Mel-frequency cepstral coefficients (MFCCs)
    MFCC = librosa.feature.mfcc(y=y,sr=sr)

    return MFCC

MFCC = extractFeatures(librosa.example("fishin"))

fig, ax = plt.subplots(nrows=1, sharex=True)
img = librosa.display.specshow(MFCC, x_axis='time', ax=ax)
fig.colorbar(img, ax=[ax])
ax.set(title='MFCC')
plt.show()

