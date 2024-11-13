import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
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


# Load data
labeled_data = pd.read_csv(GetNewestDataFileNamer('labeled'))
unlabeled_data = pd.read_csv(GetNewestDataFileNamer('unlabeled'))

def SelfLearning(labeled_data, unlabeled_data):
    complete_data_labels = labeled_data.iloc[:, 1]
    complete_data_data = labeled_data.iloc[:,2:]

    #Keep data and lables
    data = labeled_data.iloc[:, 2:].values
    labels = labeled_data.iloc[:, 1].values

    unlabeled = unlabeled_data.drop((['Filename', 'Label']), axis=1)

    # Standardize the data
    complete_data_data = StandardScaler().fit_transform(complete_data_data)
    X = StandardScaler().fit_transform(unlabeled)
    Y = StandardScaler().fit_transform(data)

    # Create a SelfTrainingClassifier with a base estimator
    base_estimator = KNeighborsClassifier(n_neighbors=5)
    self_training_model = SelfTrainingClassifier(base_estimator=base_estimator)

    # Fit the model
    self_training_model.fit(Y, labels)

    # Predict labels for the unlabeled data
    predicted_labels = self_training_model.predict(X)
    print(predicted_labels)

    # Evaluate the model on the labeled data
    y_pred = self_training_model.predict(Y)
    accuracy = accuracy_score(labels, y_pred)
    print(f'Accuracy: {accuracy}')


    # Plot the KNN classification results using PCA for visualization


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.vstack((Y, X)))
    complete_data_data_pca = pca.transform(complete_data_data)

    # Encode labels as colors

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_predicted_labels = label_encoder.transform(predicted_labels)

    # Plot the combined labels in the same figure
    plt.figure(figsize=(10, 10))

    # Plot the predicted labels for the unlabeled data
    plt.scatter(X_pca[len(Y):, 0], X_pca[len(Y):, 1], c=encoded_predicted_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
    plt.title('Predicted SelfLearning Labels for Unlabeled Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')


    plt.tight_layout()
    plt.show()
    
SelfLearning(labeled_data, unlabeled_data)