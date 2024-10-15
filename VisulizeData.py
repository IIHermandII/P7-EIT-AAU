# import matplotlib.pyplot as plt
# import librosa
# import numpy as np

# y, sr = librosa.load('SoundExample.wav',sr=None)        #f(t)

# D = np.abs(librosa.stft(y))**2                          #|F(w)|^2
# S = librosa.feature.melspectrogram(S=D, sr=sr)          # y scale --> mel scale
# fig, ax = plt.subplots()
# S_dB = librosa.power_to_db(S, ref=np.max)               # power --> db power

# img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr, ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# plt.axvline(x = 3, color = 'lime', label = 'axvline - full height')
# plt.text(3, 256, "vo", size=10, rotation=0,ha="center", va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
# plt.show()

import matplotlib.pyplot as plt
import librosa
import numpy as np
from matplotlib.widgets import Slider
from matplotlib import colormaps
import matplotlib.pyplot as mpl

# Load audio file
y, sr = librosa.load('SoundExample2.wav', sr=None)

# Compute the short-time Fourier transform and the mel spectrogram
D = np.abs(librosa.stft(y))**2
S = librosa.feature.melspectrogram(S=D, sr=sr)

# Create a figure and axis
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)

# Display the spectrogram
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

# Set title and additional annotations
plt.axvline(x=3, color='lime', label='axvline - full height')
plt.text(3, 256, "vo", size=10, rotation=0,ha="center", va="center",bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))
plt.xlabel("Time [s]")
plt.ylabel("Frequensy [Hz]")
plt.title("Mel-frequency spectrogram")
#   SLIDER DEFINED 
ax.set_xlim(0, 5)       # Start of liter
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03])    # [left, bottom, width, height]
slider = Slider(ax_slider, 'Time', 0, 18, valinit=5, valstep=0.1) 
def update(val):        # Update function for the slider
    xlim_start = slider.val
    xlim_end = xlim_start + 6  # Fixed window of 6 seconds
    if xlim_end > 24:
        xlim_start = 24 - 6  # Lock to the last 6 seconds
    ax.set_xlim(xlim_start, xlim_start + 6)
    plt.draw()
slider.on_changed(update) # Connect the slider to the update function
def on_scroll(event):   # Function to handle scroll events (optional)
    if event.button == 'up':
        slider.set_val(slider.val + 1)
    elif event.button == 'down':
        slider.set_val(slider.val - 1)
fig.canvas.mpl_connect('scroll_event', on_scroll) # Connect the scroll event to the figure

plt.show()


