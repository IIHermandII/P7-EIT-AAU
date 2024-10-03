import librosa
import soundfile as sf

# Create a simple sine wave as a test audio signal
import numpy as np
sr = 22050  # Sample rate
t = np.linspace(0, 1, sr)
x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Save to a file
sf.write('test.wav', x, sr)

# Load the file and check if it works
y, sr_loaded = librosa.load('test.wav', sr=None)
print("Loaded audio with sample rate:", sr_loaded)