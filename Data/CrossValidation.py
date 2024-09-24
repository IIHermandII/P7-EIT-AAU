from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the labeled features dataset
labels_features = pd.read_csv('Data/merged_training_dataset.csv')

# labels_features is the labeled dataset
x = labels_features.drop(['Filename', 'Label'], axis=1)
y = labels_features['Label']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform x-fold cross-validation
cv_scores = cross_val_score(rf_classifier, x, y, cv=7, scoring='accuracy')

# Print the accuracy for each fold
print("Accuracy scores for each fold:", cv_scores)

# Compute the mean and standard deviation of the cross-validation scores
print("Mean cross-validation accuracy:", cv_scores.mean())
print("Standard deviation of cross-validation scores:", cv_scores.std())


