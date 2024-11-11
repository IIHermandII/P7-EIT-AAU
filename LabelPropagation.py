from GetData import GetNewestDataFileNamer
import pandas as pd
from sklearn.semi_supervised import LabelPropagation
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from SelfLearning import SelfLearning


def LabelPropagation(labeled_data, unlabeled_data):

    labels = labeled_data.iloc[:, 1]
    dataLabeled = labeled_data.iloc[:,2:]
    dataUnlabeled = unlabeled_data.iloc[:,2:]

    # Standardize the data
    dataLabeled = StandardScaler().fit_transform(dataLabeled)
    dataUnlabeled = StandardScaler().fit_transform(dataUnlabeled)

    # Create a LabelPropagation model
    label_prop_model = LabelPropagation()
    label_prop_model.fit(dataLabeled, labels)

    # Predict labels for the unlabeled data
    predicted_labels = label_prop_model.predict(dataUnlabeled)
    print(predicted_labels)

    # Evaluate the model on the labeled data
    y_pred = label_prop_model.predict(dataLabeled)
    accuracy = accuracy_score(labels, y_pred)
    print(f'Accuracy: {accuracy}')

    # Encode labels as colors
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    encoded_predicted_labels = label_encoder.transform(predicted_labels)

    # Plot the KNN classification results using PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(np.vstack((dataLabeled, dataUnlabeled)))
    dataLabeled_pca = pca.transform(dataLabeled)
    # Plot the combined labels in the same figure
    plt.figure(figsize=(12, 6))

    # Plot the labeled data
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:len(labels), 0], X_pca[:len(labels), 1], c=encoded_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
    plt.title('Labeled Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Plot the predicted labels for the unlabeled data
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[len(labels):, 0], X_pca[len(labels):, 1], c=encoded_predicted_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
    plt.title('Predicted Labels for Unlabeled Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.tight_layout()
    plt.show()

def main():
    labeled_data = pd.read_csv(GetNewestDataFileNamer('labeled'))
    unlabeled_data = pd.read_csv(GetNewestDataFileNamer('unlabeled'))
    LabelPropagation(labeled_data, unlabeled_data)
    SelfLearning(labeled_data, unlabeled_data)
