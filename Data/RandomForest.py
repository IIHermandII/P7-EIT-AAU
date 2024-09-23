import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the labeled training data
labels_features = pd.read_csv('labels_features_importance.csv')

#  Load the validation set
validation_labels = pd.read_csv('validation_labels_importance.csv')

#  Load the dataset to be labeled
first_training = pd.read_csv('first_training_importance.csv')

#  Prepare Data for Training and Validation
# For the labeled dataset
X_train = labels_features.drop(['Filename', 'Label'], axis=1)
y_train = labels_features['Label']

# For the validation set
X_test = validation_labels.drop(['Filename', 'Label'], axis=1)
y_test = validation_labels['Label']


#  Train the Random Forest Model
# Initialize the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with labeled dataset
rf_classifier.fit(X_train, y_train)


# Make predictions on the validation set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Accuracy on Validation Set:", accuracy_score(y_test, y_pred))
print("Classification Report on Validation Set:")
print(classification_report(y_test, y_pred))

#  Label the Unlabeled Dataset
# Prepare the unlabeled dataset
X_unlabeled = first_training.drop('Filename', axis=1)

# Predict labels for the unlabeled dataset
predicted_labels = rf_classifier.predict(X_unlabeled)

# Add the predicted labels to the unlabeled dataframe
first_training['Predicted_Label'] = predicted_labels

# Save the dataframe with predicted labels
first_training.to_csv('labeled_first_training_final.csv', index=False)


# Visualize feature importance
feature_importances = rf_classifier.feature_importances_

# Feature names are the columns in X_train
feature_names = X_train.columns

# Create a pandas Series to view the feature importances with their corresponding feature names
importances = pd.Series(feature_importances, index=feature_names)

# Sort the features by importance
sorted_importances = importances.sort_values(ascending=False)


plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_importances, y=sorted_importances.index)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()