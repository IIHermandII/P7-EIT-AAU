# Branch : JU_work_02

### Classify audio.py
This script uses a trained model by the pipeline.py script to classify an audio file.
The script also allows for modifications to the audio depending on class.
The current script attenuates 'BI', 'BO' and 'M' by 12dB.
The scipt can also output the clasified segments in the original data format
and can also output the predections in a predictions.txt file.
The script output the processed audio

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive)
- Libraries: Os, Librosa, Pandas, TQDM, Numpy, Scipy, Joblib, Sourndfile, Warnings

### CSVDataVisualizer.py
The script uses PCA and LDA to plot the data in 2D.
The scipt is also used to find potential misclassifications in the data (manual),
and locate them in the data file.
The script also creates a screeplot hinting at the required dimension of the data.

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive) 
- Libraries: Os, Numpy, Re, Pandas, Scikit-learn, Matplotlib

### FeatureSelection.py
This script experiments with different feature selection schemes:
RFE(CV), Kbest, Variance thresholding and PCA analysis.

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive)
- Libraries: Os, Numpy, Re, Pandas, Scikit-learn, Matplotlib

### LabelClipsToCSV.py
This script converts the many handlabelled clips from exported from Audacity to a CSV data file.
The script uses Pandas dataframes to achive this.

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive) 
- Libraries: Os, Numpy, Re, Librosa, Pandas, Matplotlib, Datetime, Scipy, TQDM

### MergeCSV.py
This script merges the handlablled data set with a model labelled dataset (self label).

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive) 
- Libraries: Os, Re, Pandas

### Pipeline.py
This script excutes a Scikit-learn pipeline. The pipeline consists of:
StandardScalar->RFECV->Model
The result is a model trained on the best performing feature set (accuracy - cross validation).
The model is exported using joblib for use in Classify audio script to avoid training the model several times.

**Dependensis** 
- envP7RootDir (environment variable - root directory of OneDrive) 
- Libraries: Os, Numpy, Re, Librosa, Pandas, Matplotlib, Scikit-learn, Time, Warnings, Joblib

### File structure
We have the following file structure to allow the code to execute (Windows tree command):
Root: ⟵ envP7RootDir points here
+---Courses
¦   +---PAD
¦   +---Performance and Reliability Analyses of Communication Networks
¦   +---Signal processering
+---Data
¦   +---Behandlet data af Nicoleta
¦   +---CSV files
¦   +---CSV files self
¦   +---Label clips
¦   ¦   +---10
¦   ¦   +---11
¦   ¦   +---12
¦   ¦   +---13
¦   ¦   +---14
¦   ¦   +---15
¦   ¦   +---16
¦   ¦   +---17
¦   ¦   +---18
¦   ¦   +---19
¦   ¦   +---2
¦   ¦   +---20
¦   ¦   +---21
¦   ¦   +---22
¦   ¦   +---23
¦   ¦   +---3
¦   ¦   +---4
¦   ¦   +---5
¦   ¦   +---6
¦   ¦   +---7
¦   ¦   +---8
¦   ¦   +---9
¦   +---Original
¦   ¦   +---Audio and Video Files
¦   ¦   +---Video
¦   +---Processed audio
¦   +---Refined data
¦   ¦   +---Unlabeled data
¦   ¦       +---16
¦   ¦       +---17
¦   ¦       +---2
¦   ¦       +---3
¦   ¦       +---6
¦   ¦       +---7
¦   ¦       +---PROCESSED DATA
¦   ¦       +---Unlabeled test data
¦   +---Total datasets
+---Illustrations
+---Matlab
+---Sources
    +---Behandlet data af Nicoleta
    ¦   +---Data
    +---Books

Inside the label clips folders are all the clips exported from audacity files located in Refined audio.