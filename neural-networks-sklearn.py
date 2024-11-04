from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
#import seaborn as sns
import numpy as np


def GetNewestDataFileName():
    #Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    #Enter CSV directory
    workDir = envP7RootDir + "\\Data\\CSV files"
    if not os.path.exists(workDir):
        raise ValueError('Path not found (!path)')
    
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
    newestData = GetNewestDataFileName()

    # Load merged dataset
    dataTemp = pd.read_csv(newestData)

    # Remove filename column
    X = dataTemp.drop(['Filename'], axis = 1)

    # Keep data and lables 
    data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values

    print("Data length: ", len(data))

    data_S = StandardScaler().fit_transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data_S, labels, test_size = 0.2, random_state=420)

    neuralNet = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(9, 7, 3), random_state=1, max_iter=230)
    neuralNet.fit(x_train, y_train)

    nnPredictions = neuralNet.predict(x_test)
    accuracy = np.sum(nnPredictions == y_test)/len(y_test)
    print("Accuracy of model (SKlearn NN):",round(accuracy*100,3),"%")

    cm = confusion_matrix(nnPredictions, y_test)

    list = ['BI','BO','CT','M','V']
    displayCm = ConfusionMatrixDisplay(cm, display_labels=list)
    displayCm.plot(cmap = "Blues")

if __name__ == "__main__":
     main()
