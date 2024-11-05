import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
import os
import re
from sklearn.preprocessing import StandardScaler

# Make sure you are in the correct directory
def GetNewestDataFileNamer():
    # Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the application\n---> After you have made an env\n---> for vscode to see it!!")
        print("---> You need to make an env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    # Enter CSV directory, change the directory for labeled data and unlabeled data
    labeledWorkDir = envP7RootDir + "\\Data\\CSV files"
    unlabeledWorkDir = envP7RootDir + "\\Data\\Refined data\\Unlabeled data\\PROCESSED DATA"
    
    # Find all dates from the files
    labeleddirDates = []
    unlabeleddirDates = []
    for file in os.listdir(labeledWorkDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "").replace("_", "").strip("[]'")
        labeleddirDates.append([int(new_string), file])
    
    for file in os.listdir(unlabeledWorkDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "").replace("_", "").strip("[]'")
        unlabeleddirDates.append([int(new_string), file])
    
    # Sort dates and return newest
    labeleddirDates = sorted(labeleddirDates, key=lambda l: l[0], reverse=True) # Take newest data first
    unlabeleddirDates = sorted(unlabeleddirDates, key=lambda l: l[0], reverse=True) # Take newest data first

    return (labeledWorkDir + "\\" + labeleddirDates[0][1]), (unlabeledWorkDir + "\\" + unlabeleddirDates[0][1])

def PrepareData(labeledData, unlabeledData):
    LD = pd.read_csv(labeledData)
    ULD = pd.read_csv(unlabeledData)

    # Drop the first column
    LD = LD.drop(['Filename'], axis=1)
    ULD = ULD.drop(['Filename'], axis=1)

    labeledData = LD.iloc[:, 1:].values
    labeledlabels = LD.iloc[:, 0].values

    unlabeledData = ULD.iloc[:, 1:].values
    unlabeledlabels = ULD.iloc[:, 0].values

    # Scale the data to have 0 mean and variance 1 - recommended step by sklearn
    X = StandardScaler().fit_transform(labeledData)
    Y = StandardScaler().fit_transform(unlabeledData)

    # Combine labeled and unlabeled data
    X_combined = np.vstack((X, Y))
    y_combined = np.concatenate((labeledlabels, -1 * np.ones(len(unlabeledlabels))))

    # Apply LabelSpreading
    label_spread = LabelSpreading(kernel='knn', alpha=0.8)
    label_spread.fit(X_combined, y_combined)

    # Predict the labels for the unlabeled data
    y_unlabeled_pred = label_spread.transduction_[-len(unlabeledlabels):]

    print("Predicted labels for unlabeled data:")
    print(y_unlabeled_pred)

def main():
    labeledData, unlabeledData = GetNewestDataFileNamer()
    print(f"Labeled Data: {labeledData}")
    print(f"Unlabeled Data: {unlabeledData}")
    PrepareData(labeledData, unlabeledData)

if __name__ == "__main__":
    main()
