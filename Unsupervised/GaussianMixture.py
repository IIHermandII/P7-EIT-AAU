import os
import re
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

labeledData = pd.read_csv(GetNewestDataFileNamer('unlabeled'))

labels = labeledData.iloc[:, 1]
dataLabeled = labeledData.iloc[:,2:]


# Scale the data to have 0 mean and variance 1 - recommended step by sklearn
X = StandardScaler().fit_transform(dataLabeled)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=10).fit(dataLabeled)
X = pca.transform(dataLabeled)

# Create a KMeans model
Gaussian = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
Gaussian.fit(X)

# Predict the labels for the data
y_pred = Gaussian.predict(X)


# Plot the clustered data
plt.figure(figsize=(6, 6))
plt.title("Gaussian Data", size=18)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
