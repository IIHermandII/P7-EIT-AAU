import os
import re
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score, calinski_harabasz_score


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

labeledData = pd.read_csv(GetNewestDataFileNamer('labeled'))

labels = labeledData.iloc[:, 1]
dataLabeled = labeledData.iloc[:,2:]


# Scale the data to have 0 mean and variance 1 - recommended step by sklearn
X = StandardScaler().fit_transform(dataLabeled)

for n_components in [2, 5, 10, 20]:
    # Apply PCA to reduce the dimensionality of the data
    pca = PCA(n_components=n_components).fit(dataLabeled)
    X_pca = pca.transform(dataLabeled)

    # Create a KMeans model
    kmeans = cluster.KMeans(n_clusters=5, random_state=0)
    kmeans.fit(X_pca)

    # Predict the labels for the data
    y_pred = kmeans.predict(X_pca)

    # Compute the silhouette score
    sil_score = silhouette_score(X_pca, y_pred)
    print(f'Silhouette Score with PCA={n_components}: {sil_score}')

    # Compute the Calinski-Harabasz score
    ch_score = calinski_harabasz_score(X_pca, y_pred)
    print(f'Calinski-Harabasz Score with PCA={n_components}: {ch_score}')

    # Compute the Davies-Bouldin score
    db_score = davies_bouldin_score(X_pca, y_pred)
    print(f'Davies-Bouldin Score with PCA={n_components}: {db_score}')

    # Compute the Normalized Mutual Information score
    nmi_score = normalized_mutual_info_score(labels, y_pred)
    print(f'Normalized Mutual Information Score with PCA={n_components}: {nmi_score}')

    # Compute the Adjusted Rand Index
    ari_score = adjusted_rand_score(labels, y_pred)
    print(f'Adjusted Rand Index with PCA={n_components}: {ari_score}')
    print("\n")



# # Plot the clustered data
# plt.figure(figsize=(6, 6))
# plt.title("KMeans", size=18)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()
