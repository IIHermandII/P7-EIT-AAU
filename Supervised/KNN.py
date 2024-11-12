import os
import re
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

NewestDataFileName = GetNewestDataFileNamer('labeled')
print(NewestDataFileName)

from sklearn.decomposition import PCA
# Load your data
labeled_data = pd.read_csv(NewestDataFileName)  # Replace with your data file path

# Remove filename column
labels = labeled_data.iloc[:, 1]
dataLabeled = labeled_data.iloc[:,2:]

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# Scale the data to have 0 mean and variance 1 - recommended step by sklearn
X = StandardScaler().fit_transform(dataLabeled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, stratify=labels, random_state=0)

# Create a KNN classifier
clf = KNeighborsClassifier(n_neighbors=15)

# Fit the model
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Plot the KNN classification results using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the true labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('True Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot the predicted labels
# Transform the test set using the same PCA transformation
X_test_pca = pca.transform(X_test)
plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('KNN Predicted Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()