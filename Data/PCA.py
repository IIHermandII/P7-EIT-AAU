import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('first_training_importance.csv')

# The first column ('Filename') is not part of the features to be analyzed
X = df.drop('Filename', axis=1)

# Standardizing the features (important for PCA)
X_std = StandardScaler().fit_transform(X)

# Initialize PCA and fit data
pca = PCA()
principalComponents = pca.fit_transform(X_std)

# Convert to DataFrame
principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(len(X.columns))])

# Create a biplot
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='gray') # Plotting the scaled scores
    for i in range(len(coeff)):
        # Plot arrows and labels for each variable
        x, y = coeff[i,0], coeff[i,1]
        plt.arrow(0, 0, x, y, color='red', alpha=0.5)
        if labels is None:
            plt.text(x* 1.15, y * 1.15, f"Var{i+1}", color='green', ha='center', va='center')
        else:
            plt.text(x* 1.15, y * 1.15, labels[i], color='green', ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

# Create the figure for the biplot
plt.figure(figsize=(12, 8))
biplot(principalComponents[:, :2], np.transpose(pca.components_[0:2, :]), labels=X.columns)
plt.title('Biplot of first two Principal Components')
plt.show()


