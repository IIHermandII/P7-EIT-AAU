Overview
This folder contains all the necessary 
scripts and data used for training various machine learning models to analyze audio recordings from the oceanarium. The main components of the folder are detailed below.

Folder Structure

Worksheets: This file provide detailed documentation and analysis of the project, 
including methodology, results, and conclusions.

/Audio and Video Files/

Contains all the recorded audio files from the oceanarium used for segmentation and 
the 4 videos used in the experiment, each with different attached sound tracks.

/CSV Files/

CSV files resulting from different feature extractions.
ExperimentalData.csv: Contains the data collected during the experiment.


Scripts

BoostDT.py: Script for training the Gradient Boosted Decision Tree (GBDT) model.

CrossValidation.py: Script for cross-validation of each model with different folds (4, 7, and 10).

CrossValidationFinal.py: Script for cross-validation of the final model after the entire dataset was merged.

FeatureExtraction1.py: Script for extracting all features from the audio segments.

FeatureExtraction2.py: Script for extracting a reduced set of features from the audio segments.

LowerAndMute.py: Script for lowering and muting the breathing sounds of the diver.

LowerVolumeLevel.py: Script for lowering the volume level of the diver's breathing sounds.

MuteSound.py: Script for muting the diver's breathing sounds.

mergeCSV.py: Script for merging two different CSV files containing features extracted from 1-second segments.

PCA.py: Script for performing Principal Component Analysis (PCA) and creating a biplot to evaluate feature redundancy and reduce the feature set.

RandomForest.py: Script for training the Random Forest model on smaller datasets.

RandomForestFinal.py: Script for training the Random Forest model on the merged dataset.

SegmentData.py: Script for segmenting the audio files into 1-second segments.

Statistics.py: Script for data analysis.

SupportVectorMachine.py: Script for training a Support Vector Machine (SVM) model and comparing its performance against the Random Forest and GBDT models.

VideoMaker.py: Script for removing the original sound from the recorded videos and attaching the filtered audio files.