import numpy as np
import matplotlib.pyplot as plt
from maad import sound, features, rois
from maad.util import power2dB, plot2d, format_features, overlay_rois
import os
import pandas as pd
import numpy as np

# Load the audio file
s_talk, fs_talk = sound.load(r'P7-EIT-AAU/22_with_labels_talk.wav')
s_in, fs_in = sound.load(r'P7-EIT-AAU/22_with_labels_Breathe_in.wav')
s_out, fs_out = sound.load(r'P7-EIT-AAU/22_with_labels_Breathe_out.wav')

import matplotlib.pyplot as plt

# Load the audio file
s_talk, fs_talk = sound.load(r'P7-EIT-AAU/22_with_labels_talk.wav')
s_in, fs_in = sound.load(r'P7-EIT-AAU/22_with_labels_Breathe_in.wav')
s_out, fs_out = sound.load(r'P7-EIT-AAU/22_with_labels_Breathe_out.wav')

db_max = 70  # used to define the range of the spectrogram

# Process and plot the talk signal
Sxx_talk, tn_talk, fn_talk, ext_talk = sound.spectrogram(s_talk, fs_talk, nperseg=1024, noverlap=512)
Sxx_db_talk = power2dB(Sxx_talk, db_range=db_max) + db_max
plot2d(Sxx_db_talk, **{'extent': ext_talk})


# Process and plot the breathe in signal
Sxx_in, tn_in, fn_in, ext_in = sound.spectrogram(s_in, fs_in, nperseg=1024, noverlap=512)
Sxx_db_in = power2dB(Sxx_in, db_range=db_max) + db_max
plot2d(Sxx_db_in, **{'extent': ext_in})


# Process and plot the breathe out signal
Sxx_out, tn_out, fn_out, ext_out = sound.spectrogram(s_out, fs_out, nperseg=1024, noverlap=512)
Sxx_db_out = power2dB(Sxx_out, db_range=db_max) + db_max
plot2d(Sxx_db_out, **{'extent': ext_out})
plt.show()



