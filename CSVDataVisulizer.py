import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re   # Import the regular expression module
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def GetNewestDataFileNamer():
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

def ReadCVS(NewestDataFileName):
    #Read CSV file
    df = pd.read_csv(NewestDataFileName)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values

    #Scale the data to have 0 mean and variance 1 - recommended step by sklearn
    data_S = StandardScaler().fit_transform(data)

    #Fit PCA and LDA to the data
    pca = PCA(n_components=2).fit(data_S)
    lda = LDA(n_components=2).fit(data_S,labels)

    print("PCA explained var: ",sum(pca.explained_variance_ratio_))
    print("LDA explained var: ",sum(lda.explained_variance_ratio_))

    #Transform the data
    dataPCA = pca.transform(data_S)
    dataLDA = lda.transform(data_S)

    #Plot the data (raw(MFCC1,MFCC2)?, PCA, LDA)
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        plt.scatter(dataPCA[index,0],dataPCA[index,1],s=6)
    plt.title("PCA")
    plt.legend(list)

    plt.subplot(1,3,2)
    for i in list:
        index = labels == i
        plt.scatter(dataLDA[index,0],dataLDA[index,1],s=6)
    plt.title("LDA")
    plt.legend(list)

    plt.subplot(1,3,3)
    for i in list:
        index = labels == i
        plt.scatter(data[index,0],data[index,1],s=6)
    plt.title("Raw Data")
    plt.legend(list)

    #Find errors + locate index in csv file
    #Find all indencies matching label
    index = labels == "M"
    lineCSV = np.nonzero(index)
    #Remove numpy bullshit from list
    lineCSV = lineCSV[0]
    #Extract all voice clips
    test = data[lineCSV,0]
    #Find all datapoints matching the search criteria
    errors = []
    for i in range(len(test)):
        if (test[i] < -380):
            errors.append(lineCSV[i]+2)#Correct for removal of header line and start at 0 in code
    print("Potential misclassification (CSV line): ",errors)

def Nicoletta(file):
    df = pd.read_csv(file)
    X = df.drop(['Filename'], axis=1)
    a = X.iloc[:, :-1].values  # The last column is the label
    b = X.iloc[:, -1].values

    a_S = StandardScaler().fit_transform(a)
    pca = PCA(n_components=2).fit(a_S)
    lda = LDA(n_components=2).fit(a_S,b)
    dataPCA = pca.transform(a_S)
    dataLDA = lda.transform(a_S)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    list = ['breathing','noise','mixed','voice']
    for i in list:
        index = b == i
        plt.scatter(dataPCA[index,0],dataPCA[index,1],s=2)
    plt.title("PCA")
    plt.legend(list)

    plt.subplot(1,3,2)
    for i in list:
        index = b == i
        plt.scatter(dataLDA[index,0],dataLDA[index,1],s=2)
    plt.title("LDA")
    plt.legend(list)

    plt.subplot(1,3,3)
    for i in list:
        index = b == i
        plt.scatter(a[index,0],a[index,1],s=2)
    plt.title("Raw Data")
    plt.legend(list)
    print("PCA Variance " , sum(pca.explained_variance_ratio_))

def screePlot(file,dim):
    #Read CSV file
    df = pd.read_csv(file)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    data = X.iloc[:, 1:].values

    pca = PCA(n_components=dim).fit(data)
    PC_values = np.arange(pca.n_components_) + 1
    plt.figure()
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')

def main():
   NewestDataFileName = GetNewestDataFileNamer() 
   print(NewestDataFileName)
   ReadCVS(NewestDataFileName)
   #Nicoletta("labeled_first_training_final.csv")
   screePlot(NewestDataFileName,20)
   plt.show()

if __name__ == "__main__":
    main()