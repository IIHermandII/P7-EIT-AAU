import os
import re
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder


def GetNewestDataFileNamer(x):
    # Check for env variable - error if not present
    envP7RootDir = os.getenv("P7RootDir")
    if envP7RootDir is None:
        print("---> If you are working in vscode\n---> you need to restart the aplication\n---> After you have made a env\n---> for vscode to see it!!")
        print("---> You need to make a env called 'P7RootDir' containing the path to P7 root dir")
        raise ValueError('Environment variable not found (!env)')

    if x == 'labeled':
        # Enter CSV directory, change the directory for labeled data and unlabeled data
        workDir = envP7RootDir + "\\Data\\CSV files"
    else:
        workDir = envP7RootDir + "\\Data\\Refined data\\Unlabeled data\\PROCESSED DATA"

    # Find all dates from the files
    dirDates = []
    for file in os.listdir(workDir):
        onlyDate = re.findall(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', file)
        new_string = str(onlyDate).replace("-", "")
        new_string = new_string.replace("_", "")
        new_string = new_string.strip("[]'")
        dirDates.append([int(new_string), file])

    # Sort dates and return newest
    dirDates = sorted(dirDates, key=lambda l: l[1], reverse=True)  # Take oldest data first i belive
    return (workDir + "\\" + dirDates[0][1])


labeledData = pd.read_csv(GetNewestDataFileNamer('unlabeled'))

labels = labeledData.iloc[:, 1]
dataLabeled = labeledData.iloc[:, 2:]

# Encode the labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Scale the data to have 0 mean and variance 1 - recommended step by sklearn
X = StandardScaler().fit_transform(dataLabeled)

# Reduce the dimension to 10 using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Define the range of clusters
clusters = list(range(2, 21))

# Create a PdfPages object to save the plots
with PdfPages('KMeansConfusionMatrices.pdf') as pdf:
    for n_clusters in clusters:
        start_time = time.time()
        Kmeans = cluster.KMeans(n_clusters=n_clusters)
        Kmeans.fit(X_pca)

        # Predict the labels for the data
        y_pred = Kmeans.predict(X_pca)

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(encoded_labels, y_pred)

        end_time = time.time()
        print(f'Time taken for {n_clusters} clusters: {end_time - start_time:.2f} seconds')

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        ax.set_title(f'Confusion Matrix for {n_clusters} Clusters')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        tick_marks = range(n_clusters)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        pdf.savefig(fig)  # Save the confusion matrix plot into the pdf
        plt.close(fig)  # Close the figure to free memory
