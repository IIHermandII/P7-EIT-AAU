import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# Load data
complete_data = pd.read_csv('P7-EIT-AAU\\Corrected.csv')

complete_data_labels = complete_data.iloc[:, 1]
complete_data_data = complete_data.iloc[:,2:]

# Reduce sizse of data
labeled_data = complete_data.groupby('Label').apply(lambda x: x.sample(6)).reset_index(drop=True)

#Keep data and lables
data = labeled_data.iloc[:, 2:].values
labels = labeled_data.iloc[:, 1].values

unlabeled = complete_data.drop((['Filename', 'Label']), axis=1)

# Standardize the data
complete_data_data = StandardScaler().fit_transform(complete_data_data)
X = StandardScaler().fit_transform(unlabeled)
Y = StandardScaler().fit_transform(data)

# Create a SelfTrainingClassifier with a base estimator
base_estimator = KNeighborsClassifier(n_neighbors=5)
self_training_model = SelfTrainingClassifier(base_estimator=base_estimator)

# Fit the model
self_training_model.fit(Y, labels)

# Predict labels for the unlabeled data
predicted_labels = self_training_model.predict(X)
print(predicted_labels)

# Evaluate the model on the labeled data
y_pred = self_training_model.predict(Y)
accuracy = accuracy_score(labels, y_pred)
print(f'Accuracy: {accuracy}')


# Plot the KNN classification results using PCA for visualization


pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.vstack((Y, X)))
complete_data_data_pca = pca.transform(complete_data_data)

# Encode labels as colors

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_predicted_labels = label_encoder.transform(predicted_labels)
encocoded_complete_data_labels = label_encoder.transform(complete_data_labels)

# Plot the combined labels in the same figure
plt.figure(figsize=(18, 6))

# Plot the labeled data
plt.subplot(1, 3, 1)
plt.scatter(X_pca[:len(Y), 0], X_pca[:len(Y), 1], c=encoded_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('Labeled Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot the predicted labels for the unlabeled data
plt.subplot(1, 3, 2)
plt.scatter(X_pca[len(Y):, 0], X_pca[len(Y):, 1], c=encoded_predicted_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('Predicted Labels for Unlabeled Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot the complete data labels
plt.subplot(1, 3, 3)
plt.scatter(complete_data_data_pca[:, 0], complete_data_data_pca[:, 1], c=encocoded_complete_data_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('Complete Data Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()