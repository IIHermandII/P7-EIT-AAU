import os
import numpy as np
import re
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

def plotData2D(dataFile):
    #Read CSV file
    df = pd.read_csv(dataFile)

    #Keep data and lables
    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

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

    # #Plot the data (raw(MFCC1,MFCC2)?, PCA, LDA)
    plt.figure()
    #plt.subplot(1,2,1)
    #plt.figure()
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        plt.scatter(dataPCA[index,0],dataPCA[index,1],s=6)
    plt.title("PCA analysis")
    plt.xlabel("$PC_1$")
    plt.ylabel("$PC_2$")
    plt.legend(list)
    plt.title("PCA dimensionality reduction")
    plt.savefig('Figures\\PCA data (2D).pdf')

    plt.figure()
    #plt.subplot(1,2,2)
    for i in list:
        index = labels == i
        plt.scatter(dataLDA[index,0],dataLDA[index,1],s=6)
    plt.title("LDA analysis")
    plt.xlabel("$LDA_1$")
    plt.ylabel("$LDA_2$")
    plt.legend(list)
    plt.title("LDA dimensionality reduction")
    plt.savefig('Figures\\LDA data (2D).pdf')

    #Find errors + locate index in csv file
    #Find all indencies matching label
    index = labels == "M"
    lineCSV = np.nonzero(index)
    #Remove numpy bullshit from list
    lineCSV = lineCSV[0]
    #Extract all voice clips
    test = dataPCA[lineCSV,1]
    #Find all datapoints matching the search criteria
    errors = []
    for i in range(len(test)):
        if (test[i] < -0.5):
            errors.append(lineCSV[i]+2)#Correct for removal of header line and start at 0 in code
    print("Potential misclassification (CSV line): ",errors)

def plotData3D(dataFile):
    #Read CSV file
    df = pd.read_csv(dataFile)

    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

    #Scale the data to have 0 mean and variance 1 - recommended step by sklearn
    data_S = StandardScaler().fit_transform(data)

    #Fit PCA and LDA to the data
    pca = PCA(n_components=3).fit(data_S)
    lda = LDA(n_components=3).fit(data_S,labels)

    print("PCA explained var: ",sum(pca.explained_variance_ratio_))
    print("LDA explained var: ",sum(lda.explained_variance_ratio_))

    #Transform the data
    dataPCA = pca.transform(data_S)
    dataLDA = lda.transform(data_S)

    plt.figure()
    #plt.subplot(1,2,1)
    ax = plt.axes(projection ="3d")
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        ax.scatter3D(dataPCA[index,0], dataPCA[index,1], dataPCA[index,2], s=6)
    plt.legend(list)
    plt.title("PCA dimensionality reduction")
    plt.savefig('Figures\\PCA data (3D).pdf')

    plt.figure()
    #plt.subplot(1,2,2)
    ax = plt.axes(projection ="3d")
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        ax.scatter3D(dataLDA[index,0], dataLDA[index,1], dataLDA[index,2], s=6)
    plt.legend(list)
    plt.title("LDA dimensionality reduction")
    plt.savefig('Figures\\LDA data (3D).pdf')

def screePlot(file):
    #Read CSV file
    df = pd.read_csv(file)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    data = X.iloc[:, 1:].values

    data_S = StandardScaler().fit_transform(data)

    pca = PCA(n_components=data.shape[1]).fit(data_S)
    PC_values = np.arange(pca.n_components_) + 1
    plt.figure()
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree plot')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.savefig('Figures\\Scree plot.pdf')

    varEx = pca.explained_variance_ratio_

    variancePC = []
    for i in range(len(varEx)):
        variancePC.append(round(varEx[:i].sum(),3))
    print(variancePC)

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
    # fig = plt.figure()
    # ax = plt.axes(projection ="3d")
    # list = ['breathing','noise','mixed','voice']
    # for i in list:
    #     index = b == i
    #     ax.scatter3D(dataPCA[index,0], dataPCA[index,1], dataPCA[index,2], s=6)
    # plt.legend(list)

    plt.figure()
    plt.title("Old student data PCA/LDA")
    plt.subplot(1,2,1)
    list = ['breathing','noise','mixed','voice']
    for i in list:
        index = b == i
        plt.scatter(dataPCA[index,0],dataPCA[index,1],s=2)
    plt.title("PCA")
    plt.legend(list)

    plt.subplot(1,2,2)
    for i in list:
        index = b == i
        plt.scatter(dataLDA[index,0],dataLDA[index,1],s=2)
    plt.title("LDA")
    plt.legend(list)

    print("PCA Variance " , sum(pca.explained_variance_ratio_))

def main():
   envP7RootDir = os.getenv("P7RootDir")
   #Use this variable to select between our (handlabelled) and total (self labelled) datasets
   #Do this at your own risk...
   TotalDataFileName = envP7RootDir + "\\Data\\Total datasets\\Total data file (LR).csv"

   NewestDataFileName = GetNewestDataFileName() 
   print(NewestDataFileName)

   plotData2D(NewestDataFileName)
   plotData3D(NewestDataFileName)
   Nicoletta("Old student training final.csv")
   screePlot(NewestDataFileName)
   plt.show()

if __name__ == "__main__":
    main()