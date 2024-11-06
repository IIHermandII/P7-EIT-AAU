import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings

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
    warnings.filterwarnings("ignore")
    #Use this variable to select between our (handlabelled) and total (self labelled) datasets
    selectTotal = True

    if selectTotal:
        envP7RootDir = os.getenv("P7RootDir")
        NewestDataFileName = envP7RootDir + "\\Data\\Total datasets\\Total data file (LR).csv"
    else:
        NewestDataFileName = GetNewestDataFileName()

    print("Selected data file: ", NewestDataFileName)
    # Load the merged dataset
    df = pd.read_csv(NewestDataFileName)

    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

    print("Data loaded")
    print("Training model may take several minutes depending on selected\nNO PROGRESS BE PATIENT :)")
    model = LogisticRegression()
    pipe = Pipeline([('Scale data',StandardScaler()),
                    ('Feature selection',RFECV(estimator=model,cv = StratifiedKFold(5))),
                    ('Classification',model)])
    
    pipe.fit(data,labels)
    print("Initial model fit (all features)")

    pipe[1].feature_names_in_ = data.columns.values
    selected_features = pipe[1].get_feature_names_out()
    print("Selected features:\n",selected_features)

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

    data_reduced = data[selected_features]
    # Split the dataset into training and testing sets (80/20)
    x_train, x_test, y_train, y_test = train_test_split(data_reduced, labels, test_size=0.2, random_state=420)
    
    pipe.fit(x_train,y_train)
    print("Final model fit (selected features)")

    pipe[1].feature_names_in_ = data_reduced.columns.values
    print("Names features (reduced):\n",pipe[1].get_feature_names_out())

    pred = pipe.predict(x_test)
    print("LR model")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, pred):.4f}")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, pred))

    print("Classification report")

    cm = confusion_matrix(y_test, pred)
    #Prepare for plotting
    list = ['BI','BO','CT','M','V']
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=list)
    #Plot Confusion matrices
    cm_disp.plot(cmap = "Blues")

    print("Confusion matrix created")

    plt.show()

    # Save the model to disk
    joblib.dump(pipe, 'LR_model_trainset.sav')

    print("Model saved \nSCRIPT DONE")

if __name__ == "__main__":
    main()