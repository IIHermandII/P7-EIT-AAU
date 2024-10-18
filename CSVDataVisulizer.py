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
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Envirement Variable not fount (!env)')
    workDir = envP7RootDir + "\\Data\\CSV files"
    dirDats = []
    for file in os.listdir(workDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "")
        new_string = new_string.replace("_","")
        new_string = new_string.strip("[]'")
        dirDats.append([int(new_string),file])
    dirDats = sorted(dirDats,key=lambda l:l[1],reverse=True) # Take oldest data first i belive 
    return(workDir + "\\" + dirDats[0][1])

def ReadCVS(NewestDataFileName):
    # print("H")
    df = pd.read_csv(NewestDataFileName)
    X = df.drop(['Filename'], axis=1)
    a = X.iloc[:, 1:].values  # The last column is the label
    b = X.iloc[:, 0].values
    # print("DATA WITHOUT FILENAME: ")
    # print(X)
    # print("DATA POINTS")
    # print(a)
    # print("TARGET")
    # print(b)
    a_S = StandardScaler().fit_transform(a)
    pca = PCA(n_components=2).fit(a_S)
    lda = LDA(n_components=2).fit(a_S,b)
    dataPCA = pca.transform(a_S)
    dataLDA = lda.transform(a_S)
    #print(dataPCA[:,0])
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = b == i
        plt.scatter(dataPCA[index,0],dataPCA[index,1],s=4)
    plt.title("PCA")
    plt.legend(list)

    plt.subplot(1,3,2)
    for i in list:
        index = b == i
        plt.scatter(dataLDA[index,0],dataLDA[index,1],s=4)
    plt.title("LDA")
    plt.legend(list)

    plt.subplot(1,3,3)
    for i in list:
        index = b == i
        plt.scatter(a[index,0],a[index,1],s=4)
    plt.title("Raw Data")
    plt.legend(list)


    index = b == "V"
    test = a[index,1]
    for i in range(len(test)):
        if (test[i] < 80):
            print("Bo error at: ",i)

    # plt.show()
    print("PCA Variance " , sum(pca.explained_variance_ratio_))

def Nicoletta(file):
    df = pd.read_csv(file)
    X = df.drop(['Filename'], axis=1)
    a = X.iloc[:, :-1].values  # The last column is the label
    b = X.iloc[:, -1].values
    print("VECTOR")
    print(a)
    print("LABLE")
    print(b)

    a_S = StandardScaler().fit_transform(a)
    pca = PCA(n_components=2).fit(a_S)
    lda = LDA(n_components=2).fit(a_S,b)
    dataPCA = pca.transform(a_S)
    dataLDA = lda.transform(a_S)
    #print(dataPCA[:,0])
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


def main():
   NewestDataFileName = GetNewestDataFileNamer() 
   print(NewestDataFileName)
   ReadCVS(NewestDataFileName)
   Nicoletta("labeled_first_training_final.csv")
   plt.show()

if __name__ == "__main__":
    main()