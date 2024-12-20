import os
import numpy as np
import re
import pandas as pd
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

def OprationExplanation():
    print("\n\n\n-------------- CSV data Visuallized --------------")
    print("This file expects the following to work:")
    print("1.\tenv ( Environment variables / System variables):")
    print("\tP7RootDir : C:\\path\\to\\P7  i.g C:\\Users\\emill\\OneDrive - Aalborg Universitet\\P7 ")
    print("2.\tAll files at: P7\\Data\\Label clips. to have same structure")
    print("\t<name>-<number>-...-<number>.wav")

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

def plotData2D(RootDir, dataFile):
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

    print("2-Dimentions:")
    print("PCA explained var: ",sum(pca.explained_variance_ratio_))
    print("LDA explained var: ",sum(lda.explained_variance_ratio_))

    #Transform the data
    dataPCA = pca.transform(data_S)
    dataLDA = lda.transform(data_S)

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
    plt.savefig(RootDir + '\\Figures\\PCA data (2D).pdf',bbox_inches="tight")

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
    plt.savefig(RootDir + '\\Figures\\LDA data (2D).pdf',bbox_inches="tight")

    plt.figure()
    #plt.subplot(1,2,2)
    for i in list:
        index = labels == i
        plt.scatter(data_S[index,0],data_S[index,24],s=6)
    plt.xlabel("$LPC2$")
    plt.ylabel("$Spectal centroid$")
    plt.legend(list)
    plt.title("Selected feature plot")

    # #Find errors + locate index in csv file
    # #Find all indencies matching label
    # index = labels == "M"
    # lineCSV = np.nonzero(index)
    # #Remove numpy bullshit from list
    # lineCSV = lineCSV[0]
    # #Extract all voice clips
    # test = dataPCA[lineCSV,1]
    # #Find all datapoints matching the search criteria
    # errors = []
    # for i in range(len(test)):
    #     if (test[i] < -0.5):
    #         errors.append(lineCSV[i]+2)#Correct for removal of header line and start at 0 in code
    # print("Potential misclassification (CSV line): ",errors)

def plotData3D(RootDir, dataFile):
    #Read CSV file
    df = pd.read_csv(dataFile)

    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

    #Scale the data to have 0 mean and variance 1 - recommended step by sklearn
    data_S = StandardScaler().fit_transform(data)

    #Fit PCA and LDA to the data
    pca = PCA(n_components=3).fit(data_S)
    lda = LDA(n_components=3).fit(data_S,labels)

    print("3-Dimentions:")
    print("PCA explained var: ",sum(pca.explained_variance_ratio_))
    print("LDA explained var: ",sum(lda.explained_variance_ratio_))

    #Transform the data
    dataPCA = pca.transform(data_S)
    dataLDA = lda.transform(data_S)

    plt.figure()
    #plt.subplot(1,2,1)
    ax = plt.axes(projection ="3d")
    ax.view_init(elev=10,azim=-75,roll=0)
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        ax.scatter3D(dataPCA[index,0], dataPCA[index,1], dataPCA[index,2], s=6)
    plt.legend(list)
    plt.title("PCA dimensionality reduction")
    plt.savefig(RootDir + '\\Figures\\PCA data (3D).pdf',bbox_inches="tight")

    plt.figure()
    #plt.subplot(1,2,2)
    ax = plt.axes(projection ="3d")
    ax.view_init(elev=25,azim=140,roll=0)
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        ax.scatter3D(dataLDA[index,0], dataLDA[index,1], dataLDA[index,2], s=6)
    plt.legend(list)
    plt.title("LDA dimensionality reduction")
    plt.savefig(RootDir + '\\Figures\\LDA data (3D).pdf',bbox_inches="tight")

    plt.figure()
    #plt.subplot(1,2,2)
    ax = plt.axes(projection ="3d")
    ax.view_init(elev=25,azim=140,roll=0)
    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        ax.scatter3D(data_S[index,0], data_S[index,5], data_S[index,24], s=6)
    plt.legend(list)
    plt.title("Selected features")

def screePlot(RootDir, dataFile):
    #Read CSV file
    df = pd.read_csv(dataFile)

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
    plt.savefig(RootDir + '\\Figures\\Scree plot.pdf',bbox_inches="tight")

    varEx = pca.explained_variance_ratio_

    variancePC = []
    for i in range(len(varEx)):
        variancePC.append(round(varEx[:i].sum(),3))
    print("Variance of Principal Components [1..n]")
    print(variancePC)

    plt.figure()
    plt.plot(PC_values[:-1], variancePC[1:], 'o-', linewidth=2, color='blue')
    plt.title('Cumulative "Scree plot"')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained ratio')
    plt.savefig(RootDir + '\\Figures\\Scree plot Inv.pdf',bbox_inches="tight")

def biplot(RootDir, dataFile):
    #Read CSV file
    df = pd.read_csv(dataFile)

    data = df.drop(['Filename','Label'], axis=1)
    labels = df['Label']

    column_order = data.columns.values

    #Scale the data to have 0 mean and variance 1 - recommended step by sklearn
    data_S = StandardScaler().fit_transform(data)
    
    pca = PCA()
    dataPCA = pca.fit_transform(data_S)

    dataPCA = dataPCA[:, :2]

    plt.figure()
    # Create a biplot
    xs = dataPCA[:,0]
    ys = dataPCA[:,1]

    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())

    list = ['BI','BO','CT','V','M']
    for i in list:
        index = labels == i
        plt.scatter(xs[index]*scalex,ys[index]*scaley,s=6)
    plt.legend(list)

    coeff = pca.components_[0:2, :]
    for i in range(len(coeff[0])):
        # Plot arrows and labels for each variable
        x, y = coeff[0,i], coeff[1,i]
        plt.arrow(0, 0, x, y, color='red', alpha=0.5)
        plt.text(x* 1.1, y * 1.1, column_order[i], color='black', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
    plt.xlabel("$PC_1$")
    plt.ylabel("$PC_2$")
    plt.grid()
    plt.title('Biplot of first two principal components')
    plt.savefig(RootDir + '\\Figures\\Biplot.pdf',bbox_inches="tight")
    #plt.show()

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
    #TotalDataFileName = envP7RootDir + "\\Data\\Total datasets\\Total data file (LR).csv"
    RootDir = sys.argv[1]
    print("EL argument:", sys.argv[1])
    NewestDataFileName = GetNewestDataFileName(RootDir) 
    # NewestDataFileName = "Datasets\\OurData.csv"
    print(NewestDataFileName)
    plotData2D(RootDir, NewestDataFileName)
    plotData3D(RootDir, NewestDataFileName)
    # PredictedDataFileName = "C:\\Users\\emill\OneDrive - Aalborg Universitet\\P7\\Data\\CSV files self\\2.csv"
    # plotData3D(PredictedDataFileName)
    # Nicoletta("Datasets\\Old student training_final.csv")
    biplot(RootDir, NewestDataFileName) 
    screePlot(RootDir, NewestDataFileName)
    print(colored("Remember do close plots befor this process will continiue...", 'red', attrs=['bold']))
    plt.show()
    print(colored("continiueing now", 'green', attrs=['bold']))

if __name__ == "__main__":
    main()