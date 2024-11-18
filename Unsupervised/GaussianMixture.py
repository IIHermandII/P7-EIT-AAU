import os
import re
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score

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

fig, axs = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('GaussianMixture Metrics for PCA with different components', size=18)

# Initialize a dictionary to store the scores for each metric
scores = {
    'Silhouette Score': [],
    'Calinski-Harabasz Score': [],
    'Davies-Bouldin Score': [],
    'Normalized Mutual Information Score': [],
    'Adjusted Rand Index': []
}

components = list(range(1, 21))

for idx, n_components in enumerate(components):
    # Apply PCA to reduce the dimensionality of the data
    pca = PCA(n_components=n_components).fit(dataLabeled)
    X = pca.transform(dataLabeled)

    # Create a Gaussian Mixture model
    Gaussian = GaussianMixture(n_components=5, covariance_type="full", random_state=42)
    Gaussian.fit(X)

    # Predict the labels for the data
    y_pred = Gaussian.predict(X)

    # Compute the silhouette score
    sil_score = silhouette_score(X, y_pred)
    ch_score = calinski_harabasz_score(X, y_pred)
    db_score = davies_bouldin_score(X, y_pred)
    nmi_score = normalized_mutual_info_score(labels, y_pred)
    ari_score = adjusted_rand_score(labels, y_pred)

    # Store the scores in the dictionary
    scores['Silhouette Score'].append(sil_score)
    scores['Calinski-Harabasz Score'].append(ch_score)
    scores['Davies-Bouldin Score'].append(db_score)
    scores['Normalized Mutual Information Score'].append(nmi_score)
    scores['Adjusted Rand Index'].append(ari_score)

# Plot each metric score in a subplot
for i, (metric, score_list) in enumerate(scores.items()):
    ax = axs[i // 2, i % 2]
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xticks(components)
    ax.plot(components, score_list, marker='o', linestyle='-')
    ax.set_title(metric)
    ax.set_ylabel('Score')

# Remove the empty subplot
fig.delaxes(axs[2, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# # Plot the clustered data
# plt.figure(figsize=(6, 6))
# plt.title("GaussianMixture", size=18)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=10, cmap='viridis')
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

