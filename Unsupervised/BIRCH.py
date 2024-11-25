import os
import re
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score, calinski_harabasz_score
import time
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics.cluster import pair_confusion_matrix


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
    print(workDir + "\\" + dirDates[0][1])
    return(workDir + "\\" + dirDates[0][1])

labeledData = pd.read_csv(GetNewestDataFileNamer('labeled'))

labels = labeledData.iloc[:, 1]
dataLabeled = labeledData.iloc[:,2:]


# Scale the data to have 0 mean and variance 1 - recommended step by sklearn
X = StandardScaler().fit_transform(dataLabeled)

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('BIRCH Metrics for different K clusters', size=18)

# Initialize a dictionary to store the scores for each metric
scores = {
    'Silhouette Score': [],
    'Calinski-Harabasz Score': [],
    'Davies-Bouldin Score': [],
    'Normalized Mutual Information Score': [],
    'Adjusted Rand Index': []
}

# clusters = list(range(2, 21))
clusters = [5]

# Reduce the dimension to 10 using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

for idx, n_clusters in enumerate(clusters):
    Birch = cluster.Birch(n_clusters=n_clusters)
    Birch.fit(X_pca)
    start_time = time.time()

    # Predict the labels for the data
    y_pred = Birch.predict(X_pca)
    
    # Compute the silhouette score
    sil_score = silhouette_score(X_pca, y_pred)
    ch_score = calinski_harabasz_score(X_pca, y_pred)
    db_score = davies_bouldin_score(X_pca, y_pred)
    nmi_score = normalized_mutual_info_score(labels, y_pred)
    ari_score = adjusted_rand_score(labels, y_pred)

    # print(f'\nPCA with {n_components} components:')
    # print(f'Silhouette Score: {sil_score}')
    # print(f'Calinski-Harabasz Score: {ch_score}')
    # print(f'Davies-Bouldin Score: {db_score}')
    # print(f'Normalized Mutual Information Score: {nmi_score}')
    # print(f'Adjusted Rand Index: {ari_score}')

    # Store the scores in the dictionary
    scores['Silhouette Score'].append(sil_score)
    scores['Calinski-Harabasz Score'].append(ch_score)
    scores['Davies-Bouldin Score'].append(db_score)
    scores['Normalized Mutual Information Score'].append(nmi_score)
    scores['Adjusted Rand Index'].append(ari_score)

    end_time = time.time()
    print(f'Time taken for {n_clusters} clusters: {end_time - start_time:.2f} seconds')

# Plot each metric score in a subplot
# for i, (metric, score_list) in enumerate(scores.items()):
#     ax = axs[i // 2, i % 2]
#     ax.plot(clusters, score_list, marker='o', linestyle='-')
#     ax.set_title(metric)
#     ax.set_ylabel('Score')
#     ax.set_xticks(clusters)

# plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()

# Create a PdfPages object to save the plots
with PdfPages('BIRCHAllData.pdf') as pdf:
    # for i, (metric, score_list) in enumerate(scores.items()):
    #     fig, ax = plt.subplots()
    #     ax.plot(clusters, score_list, marker='o', linestyle='-')
    #     ax.set_title(metric, size=24)
    #     ax.set_ylabel('Score', size=20)
    #     ax.set_xticks(clusters)
    #     ax.set_xlabel('Number of Clusters', size=20)
    #     pdf.savefig(fig)  # Save the current figure into the pdf
    #     plt.close(fig)  # Close the figure to free memory

    # Save the scatter plot with cluster centers
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
    ax.set_title(f'BIRCH Clustering with {n_clusters} Clusters', size = 24)
    ax.set_xlabel('Principal Component 1', size = 20)
    ax.set_ylabel('Principal Component 2', size = 20)
    pdf.savefig(fig)  # Save the scatter plot into the pdf
    plt.close(fig)  # Close the figure to free memory

