import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# Load the labeled training data
labels_features = pd.read_csv('labels_features_importance.csv')

# Load the validation set
validation_labels = pd.read_csv('validation_labels_importance.csv')

# Load the dataset to be labeled
first_training = pd.read_csv('first_training_importance.csv')

# Prepare Data for Training and Validation
# For the labeled dataset
X_train = labels_features.drop(['Filename', 'Label'], axis=1)
y_train = labels_features['Label']

# For the validation set
X_test = validation_labels.drop(['Filename', 'Label'], axis=1)
y_test = validation_labels['Label']

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the model with the labeled dataset
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate the Model on the Validation Set
# Make predictions on the validation set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the model
print("Accuracy on Validation Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Validation Set:")
print(classification_report(y_test, y_pred))

# Label the Unlabeled Dataset
X_unlabeled_scaled = scaler.transform(first_training.drop('Filename', axis=1))

# Predict labels for the unlabeled dataset
predicted_labels = svm_classifier.predict(X_unlabeled_scaled)

# Add the predicted labels to the unlabeled dataframe
first_training['Predicted_Label'] = predicted_labels

# Save the dataframe with predicted labels
first_training.to_csv('labeled_first_training_svm.csv', index=False)



