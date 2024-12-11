# P7-EIT-AAU Diver Voice Recognition
by: Emil Leth Høimark, Jens-Ulrik Ladekjær-Mikkelsen, Casper Nørrergaard Bak, Steffen Damm, Marcus Mogensen, Magnus Høgh.
### Overview of the Project

This project was completed during the 7th semester (the first semester of the Master's program) for Electronic Engineers at Aalborg University (AAU), Denmark. The theme of the project was Machine Learning (ML), and we selected the following task:

At Nordsøen, a large indoor aquarium, divers communicate underwater during feeding sessions. While fish are being fed, the divers explain things to the audience through a microphone seen through a glass window. However, the sound of breathing, other equipment noises, and pressurized air often interfere with the diver's speech. The task was to label the sounds and differentiate between actual speech and background noise.

We successfully implemented a solution for this problem. The project is designed to be plug-and-play, easy to follow, and simple to understand. *In case you have any feedback or criticism regarding the code, we kindly refer you to the university they should have taught us better—though we are perfect in every way! (just a joke!)*

### Running the Code

Not all files comes in the repo so you need to run the code to see it all :D
To run the project, make sure to update the `main.py` file with the correct path to the "Data" folder, and then run the `main.py` script. Ensure that all dependencies are installed (e.g., via `pip`). The project is designed to run with Python 3.

### LabelClipsToCSV.py

This script converts the manually labeled clips from Audacity into a CSV data file using Pandas dataframes. The output is saved as a CSV file for further analysis.

**Dependencies**  
- Libraries: os, re, librosa, numpy, pandas, datetime, scipy.stats, tqdm, warnings, sys

### CSVDataVisualizer.py

This script uses PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) to visualize the data in 2D or 3D. It also helps to identify potential misclassifications in the data (manual inspection) and locate them within the data file. The script generates a scree plot to hint at the optimal data dimensions.

**Dependencies**  
- Libraries: os, numpy, re, pandas, matplotlib.pyplot, sklearn.preprocessing, sklearn, sys

### Pipeline.py

This script executes a Scikit-learn pipeline that builds, trains, evaluates, and exports a binary file for the machine learning model. 

**Dependencies**  
- Libraries: os, re, pandas, numpy, matplotlib.pyplot, matplotlib.lines, joblib, warnings, time, sklearn, sys

### SmartLabelAudio.py

This script uses the newly created binary model to label a sound file and save the labels to a file.

**Dependencies**  
- Libraries: os, numpy, librosa, soundfile, scipy, joblib, pandas, tqdm, warnings, termcolor, sys

### CompareMLToAudacity.py

This script compares the ML-labeled file to the manually labeled file from Audacity and evaluates the accuracy in percentage.

**Dependencies**  
- Libraries: os, sklearn, matplotlib, numpy, tqdm, seaborn, soundfile, sys, termcolor

### File Structure

The project follows this directory structure to ensure the code runs smoothly:



+---P7-EIT-AAU<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---main.py (main file calling they other files)<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---ComapareMLToAudacity.py<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---LabelClipsToCSV.py<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Pipeline.py<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---SmartLableAudio.py<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Data<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---CSV files<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Figures<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Label clips<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Models<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Outputs<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---(RESULTS)<br />
¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;¦&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+---Soundfile<br />
