import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def GetNewestDataFileName():
    #Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    #Enter CSV directory
    workDir = envP7RootDir + "\\Data\\CSV files"
    
    #Find all dates from the files
    dirDates = []
    for file in os.listdir(workDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "")
        new_string = new_string.replace("_","")
        new_string = new_string.strip("[]'")
        dirDates.append([int(new_string),file])
    
    #Sort dates and return newest
    dirDates = sorted(dirDates,key=lambda l:l[1],reverse=True) # Take oldest data first i belive 
    return(workDir + "\\" + dirDates[0][1])


def main():
    # Load the model from disk
    pipe = joblib.load('LR_model_trainset.sav')
    data_reduce = joblib.load('LR Data.csv')

    NewestDataFileName = GetNewestDataFileName() 
    # Load the merged dataset
    df = pd.read_csv(NewestDataFileName)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    #data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values

    # Split the dataset into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(data_reduce, labels, test_size=0.2, random_state=420)


    print(pipe[1].get_feature_names_out())
    
    pred = pipe.predict(x_test)
    print("LR model")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, pred):.4f}")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, pred))


if __name__ == "__main__":
    main()