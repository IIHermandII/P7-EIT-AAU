import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import joblib
import warnings
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.decomposition import PCA
import sys

def GetNewestDataFileName(RootDir):
    workDir = RootDir + "\\CSV files"
    
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
            model = svm.SVC(kernel='linear',probability=True,max_iter=1000)
        case "RF":
            model = RandomForestClassifier(random_state=420, n_jobs=-1)
        case "GBDT":
            model = GradientBoostingClassifier(random_state=420)

    return model

def selectDataset(str):
    envP7RootDir = os.getenv("P7RootDir")
    match str:
        case "Trainset":
            fileName = "Datasets\\OurData.csv"
        case "Expanded":
            fileName = envP7RootDir + "\\Data\\Total datasets\\Total data file (SVM+Confidence).csv"
        case "500Hz":
            fileName = "Datasets\\OurData 500Hz.csv"
        case "1kHz":
            fileName = "Datasets\\OurData 1kHz.csv"
        case "2kHz":
            fileName = "Datasets\\OurData 2kHz.csv"
        case "4kHz":
            fileName = "Datasets\\OurData 4kHz.csv"
        case "8kHz":
            fileName = "Datasets\\OurData 8kHz.csv"
        case "16kHz":
            fileName = "Datasets\\OurData 16kHz.csv"
        case "24kHz":
            fileName = "Datasets\\OurData 24kHz.csv"
    return fileName

def MakeModel(RootDir, data, selectRFE, model, x_train, y_train):
    print("Training...", end="\r")
    start_time = time.perf_counter()
    if selectRFE:
        pipe = Pipeline([('Scale data',StandardScaler()),
                        ('Feature selection',RFE(estimator=model,n_features_to_select=10))])
    else:
        pipe = Pipeline([('Scale data',StandardScaler()),
                        ('Feature selection',RFECV(estimator=model,cv = StratifiedKFold(4),min_features_to_select=1,n_jobs=-1))])

    pipe.fit(x_train,y_train)
    print("           ", end="\r")  # Clears the line by printing spaces
    print("Traning - DONE")
    
    print("Traning Time :\t%.3f " % (time.perf_counter() - start_time))
    pipe[1].feature_names_in_ = data.columns.values
    selected_features = pipe[1].get_feature_names_out()
    print("Selected features :\t",selected_features)

    if not selectRFE:
        #Plot results
        cv_results = pd.DataFrame(pipe[1].cv_results_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean accuracy")
        plt.errorbar(x=cv_results["n_features"],
                     y=cv_results["mean_test_score"],
                     yerr=cv_results["std_test_score"],
                     ecolor='red')
        
        # Add legend manually
        line_legend = Line2D([0], [0], color='blue', marker='', label='Data line')  # Line for the data
        error_legend = Line2D([0], [0], color='red', linestyle='', marker='|', label='Error bars')  # Error bar legend

        # Customize plot
        plt.legend(handles=[line_legend, error_legend])
        plt.title("Accuracy vs dimension")
        plt.savefig(RootDir +"\\Figures\\Features vs accuracy.pdf",bbox_inches="tight")
        print("Feature plot created")
    return pipe

def main():
    warnings.filterwarnings("ignore")
    RootDir = sys.argv[1]
    #Use these variables to select trainset and models
    selectRFE = True
    modelName = "SVM"
    model = selectModel(modelName)

    # --- DATA WORK ---
    print("Preparing data", end="\r")
    dataFile = GetNewestDataFileName(RootDir)
    df = pd.read_csv(dataFile)
    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=420) # Split the dataset into training and testing sets (80/20)
    print("                 ", end="\r")
    print("Data - DONE")

    # --- ML WORK ---
    PipelineedMLModel = MakeModel(RootDir, data, selectRFE, model, x_train, y_train)
    pred = PipelineedMLModel.predict(x_test)
    print(modelName +" model (RFE)")
    print(f"Accuracy on Test Set: {accuracy_score(y_test, pred):.4f}")
    print("Classification Report on Test Set:")
    print(classification_report(y_test, pred,digits=3))

    # --- RAPPORT WORK ---
    f = open(RootDir + "\\Models\\ " + modelName + " classification initial.txt", "w")
    f.write(classification_report(y_test, pred,digits=3))
    f.close()

    # --- CONFUSION MATRIX WORK ---
    cm = confusion_matrix(y_test, pred)
    list = ['BI','BO','CT','M','V']
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=list)
    #Plot Confusion matrices
    cm_disp.plot(cmap = "Blues")
    plt.savefig(RootDir+ "\\Figures\\" + modelName + " confusion matrix.pdf",bbox_inches="tight")
    joblib.dump(PipelineedMLModel, RootDir+ "\\Models\\" + modelName + "_model_trainset.sav")
    plt.show()

if __name__ == "__main__":
    main()



# scores = cross_val_score(pipe,data,labels,cv=4)
# print("4-fold cross validation")
# print(f"Accuracy for each fold: {scores}")
# print(f"Mean accuracy: {scores.mean():.4f}")
# print(f"Standard deviation: {scores.std():.4f}")

# scores = cross_val_score(pipe,data,labels,cv=7)
# print("7-fold cross validation")
# print(f"Accuracy for each fold: {scores}")
# print(f"Mean accuracy: {scores.mean():.4f}")
# print(f"Standard deviation: {scores.std():.4f}")

# scores = cross_val_score(pipe,data,labels,cv=10)
# print("10-fold cross validation")
# print(f"Accuracy for each fold: {scores}")
# print(f"Mean accuracy: {scores.mean():.4f}")
# print(f"Standard deviation: {scores.std():.4f}")