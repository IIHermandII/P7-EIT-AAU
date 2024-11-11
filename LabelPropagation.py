import pandas as pd
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
import os
import re


def GetNewestDataFileNamer(x):
    #Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')
    
    if x == 'labeled':
        #Enter CSV directory, change the directory for labeled data and unlabeled data
        workDir = envP7RootDir + "\\Data\\CSV files"
    else:
        workDir = envP7RootDir + "\\Data\\Refined data\\Unlabeled data\\PROCESSED DATA"
    
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


def LabelPropagation(labeled_data, unlabeled_data):

    labels = labeled_data.iloc[:, 1]
    dataLabeled = labeled_data.iloc[:,2:]
    dataUnlabeled = unlabeled_data.iloc[:,2:]


    # Standardize the data
    dataLabeled = StandardScaler().fit_transform(dataLabeled)
    dataUnlabeled = StandardScaler().fit_transform(dataUnlabeled)
    # Create a LabelSpreading model
    label_spread = LabelSpreading(kernel="knn", alpha=0.8)
    label_spread.fit(dataLabeled, labels)

    # Predict labels for the unlabeled data
    predicted_labels = label_spread.predict(dataUnlabeled)

    # Evaluate the model on the labeled data
    y_pred = label_spread.predict(dataLabeled)
    accuracy = accuracy_score(labels, y_pred)
    print(f'Accuracy: {accuracy}')

    # Encode labels as colors
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_predicted_labels = label_encoder.transform(predicted_labels)

    # Plot the KNN classification results using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.vstack((dataLabeled, dataUnlabeled)))
    dataLabeled_pca = pca.transform(dataLabeled)
    # Plot the combined labels in the same figure
    plt.figure(figsize=(12, 6))

    # Plot the labeled data
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:len(labels), 0], X_pca[:len(labels), 1], c=encoded_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
    plt.title('Label Propagation Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plot the predicted labels for the unlabeled data
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[len(labels):, 0], X_pca[len(labels):, 1], c=encoded_predicted_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
    plt.title('Predicted Labels for Unlabeled Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()


labeled_data = pd.read_csv(GetNewestDataFileNamer('labeled'))
unlabeled_data = pd.read_csv(GetNewestDataFileNamer('unlabeled'))
LabelPropagation(labeled_data, unlabeled_data)
    
