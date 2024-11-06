import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data = pd.read_csv('P7-EIT-AAU\\Corrected.csv')

# Print data
print(data)

# Save the correct labels from the labeled data
correct_labels = data.iloc[:, 0].values

# Separate labeled and unlabeled data
labeled_data = data.iloc[::10]
labeled_data = labeled_data.drop((['Filename']), axis=1)

unlabeled = data.drop((['Filename']), axis=1)
unlabeled = unlabeled.drop((['Label']), axis=1)

# Print unlabeled and labeled data
print(unlabeled)
print(labeled_data)

# Identify categorical columns
categorical_columns = labeled_data.select_dtypes(include=['object']).columns

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    labeled_data[col] = le.fit_transform(labeled_data[col])
    unlabeled[col] = le.transform(unlabeled[col])
    label_encoders[col] = le

# Keep data and labels
labeled_features = labeled_data.iloc[:, 1:].values
labeled_labels = labeled_data.iloc[:, 0].values

unlabeled_features = unlabeled.values

# Standardize the data
X = StandardScaler().fit_transform(unlabeled_features)
Y = StandardScaler().fit_transform(labeled_features)

# Create a SelfTrainingClassifier with a base estimator
base_estimator = KNeighborsClassifier(n_neighbors=5)
self_training_model = SelfTrainingClassifier(base_estimator=base_estimator)

# Fit the model
self_training_model.fit(Y, labeled_labels)

# Predict labels for the unlabeled data
predicted_labels = self_training_model.predict(X)
print(predicted_labels)

# Evaluate the model on the labeled data
y_pred = self_training_model.predict(Y)
accuracy = accuracy_score(labeled_labels, y_pred)
print(f'Accuracy: {accuracy}')

# Plot the KNN classification results using PCA for visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.vstack((Y, X)))

# Plot the combined labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:len(Y), 0], X_pca[:len(Y), 1], c=labeled_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('Labeled Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Plot the predicted labels for the unlabeled data
plt.subplot(1, 2, 2)
plt.scatter(X_pca[len(Y):, 0], X_pca[len(Y):, 1], c=predicted_labels, cmap=plt.cm.Paired, edgecolor='k', s=20)
plt.title('Predicted Labels for Unlabeled Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.show()
