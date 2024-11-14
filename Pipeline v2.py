import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.decomposition import PCA

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

def selectModel(str):
    match str:
        case "LR":
            model = LogisticRegression()
        case "SVM":
            model = svm.SVC(kernel='linear',max_iter=1000)
        case "RF":
            model = RandomForestClassifier(random_state=420, n_jobs=-1)
        case "GBDT":
            model = GradientBoostingClassifier(random_state=420)

    return model

def selectDataset(str):
    envP7RootDir = os.getenv("P7RootDir")
    match str:
        case "Trainset":
            fileName = envP7RootDir + "\\Data\\Total datasets\\Total data file (LR).csv"
        case "Expanded":
            fileName = GetNewestDataFileName()

    return fileName

def main():
    warnings.filterwarnings("ignore")
    #Use this variable to select between our (handlabelled) and total (self labelled) datasets

    selectRFE = True
    modelName = "GBDT"
    model = selectModel(modelName)

    datasetName = "Trainset"
    dataFile = selectDataset(datasetName)

    print("Selected data file: ", dataFile)
    # Load the merged dataset
    df = pd.read_csv(dataFile)

    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

    # Split the dataset into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=420)

    print("Data loaded")
    print("Training model may take several minutes depending on selected\nNO PROGRESS BAR BE PATIENT :)")
        
    if selectRFE:
        pipe = Pipeline([('Scale data',StandardScaler()),
                        ('Feature selection',RFE(estimator=model,n_features_to_select=10))])
        # pipe = Pipeline([('Scale data',StandardScaler()),
        #                   ('PCA',PCA(n_components=10)),
        #                   ('Model',model)])
    else:
        pipe = Pipeline([('Scale data',StandardScaler()),
                        ('Feature selection',RFECV(estimator=model,cv = StratifiedKFold(5),min_features_to_select=4))])

    start_time = time.perf_counter()
    pipe.fit(x_train,y_train)

    print("Fit + feature elimination took --- %.3f seconds ---" % (time.perf_counter() - start_time))

    print("Initial model fit (all features)")

    pipe[1].feature_names_in_ = data.columns.values
    selected_features = pipe[1].get_feature_names_out()
    print("Selected features:\n",selected_features)

    if not selectRFE:
        #Plot results
        cv_results = pd.DataFrame(pipe[1].cv_results_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test accuracy")
        plt.errorbar(x=cv_results["n_features"],
                     y=cv_results["mean_test_score"],
                     yerr=cv_results["std_test_score"])
        plt.title("Recursive Feature Elimination \nwith correlated features")

        print("Feature plot created")

    pred = pipe.predict(x_test)
    print(modelName +" model")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, pred):.4f}")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, pred,digits=3))

    f = open("Models\\ " + modelName + " classification (initial).txt", "w")
    f.write(classification_report(y_test, pred,digits=3))
    f.close()

    print("Classification report")

    cm = confusion_matrix(y_test, pred)
    #Prepare for plotting
    list = ['BI','BO','CT','M','V']
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=list)
    #Plot Confusion matrices
    cm_disp.plot(cmap = "Blues")

    print("Confusion matrix created")

    # Save the model to disk
    joblib.dump(pipe, "Models\\" + modelName + "_model_trainset.sav")

    plt.show()
    print("Model saved \nSCRIPT DONE")

if __name__ == "__main__":
    main()