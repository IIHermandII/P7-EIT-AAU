import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn import svm


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

def VarianceTest(features):
    #Variance
    var_features = np.var(features, axis=0)
    var_sorted = pd.Series(var_features).sort_values(ascending=False)
    print("Variance vector (sorted):\n",var_sorted)

def KbestTest(features_S,labels,column_order):
    #Using univariate feature selection
    Kbest_f = SelectKBest(f_classif, k=10).fit(features_S, labels)
    
    #Convert result from x0 x1 x2... to feature names
    result = Kbest_f.get_feature_names_out()
    result = [int(s.replace('x', '')) for s in result]
    features_out = [column_order[i] for i in result]
    #Print the selected features
    print("Selected features:\n",features_out)

    #Using univariate feature selection
    Kbest_mif = SelectKBest(mutual_info_classif, k=10).fit(features_S, labels)
    #Convert result from x0 x1 x2... to feature names
    result = Kbest_mif.get_feature_names_out()
    result = [int(s.replace('x', '')) for s in result]
    features_out = [column_order[i] for i in result]
    #Print the selected features
    print("Selected features:\n",features_out)

def RecursiveElimFeature(data_S,labels,column_order):
    #Recursive feature elimination - given model here:
    clf = LogisticRegression()
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=3,
    )

    rfecv.fit(data_S, labels)

    print(f"Optimal number of features: {rfecv.n_features_}")
    #Convert result from x0 x1 x2... to feature names
    result = rfecv.get_feature_names_out()
    result = [int(s.replace('x', '')) for s in result]
    features_out = [column_order[i] for i in result]
    #Print the selected features
    print("Selected features:\n",features_out)

    #Plot results
    cv_results = pd.DataFrame(rfecv.cv_results_)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    #plt.show()

def biplot(data_S, labels,column_order):
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
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

    #plt.show()

def main():
    ##Import data
    # Load the merged dataset   
    NewestDataFileName = GetNewestDataFileName() 
    df = pd.read_csv(NewestDataFileName)

    #Remove filename coloum
    X = df.drop(['Filename'], axis=1)

    #Keep data and lables
    data = X.iloc[:, 1:].values
    labels = X.iloc[:, 0].values

    data_S = StandardScaler().fit_transform(data)

    #Remove filename coloum
    features = df.drop(['Filename','Label'], axis=1)

    features_S = StandardScaler().fit_transform(features)
    column_order = [f'LPC{i + 2}' for i in range(5)] + [f'MFCC{i + 1}' for i in range(13)] + \
                   ['MFCC_Var', 'Spectral_Contrast_Mean', 'Spectral_Contrast_Var', 
                    'SFM', 'Spectral_Spread', 'Spectral_Skewness', 'Spectral_Centroid', 
                    'Chroma_Mean', 'Chroma_Var', 'ZCR', 'STE', 'RMS']
    
    ##Feature selection tests
    VarianceTest(features)

    KbestTest(features_S,labels,column_order)

    RecursiveElimFeature(data_S,labels,column_order)

    biplot(data_S,labels,column_order)
    plt.show()

if __name__ == "__main__":
    main()