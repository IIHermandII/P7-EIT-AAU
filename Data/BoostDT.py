import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#  Load the labeled training data
labels_features = pd.read_csv('labeled_first_training_rf_imp.csv')

#  Load the validation set
validation_labels = pd.read_csv('validation_labels_importance.csv')

#  Load the dataset to be labeled
first_training = pd.read_csv('second_training.csv')

#  Prepare Data for Training and Validation
# For the labeled dataset
X_train = labels_features.drop(['Filename', 'Label'], axis=1)
y_train = labels_features['Label']

# For the validation set
X_test = validation_labels.drop(['Filename', 'Label'], axis=1)
y_test = validation_labels['Label']

# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Train the model on the labeled data
gb_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_gb = gb_classifier.predict(X_test)

# Evaluate the model
print("Accuracy on Validation Set:", accuracy_score(y_test, y_pred_gb))
print("Classification Report on Validation Set:")
print(classification_report(y_test, y_pred_gb))

X_unlabeled = first_training.drop('Filename', axis=1)

# Predict labels for the unlabeled dataset
predicted_labels_gb = gb_classifier.predict(X_unlabeled)

# Add the predicted labels back to the original DataFrame
first_training['Predicted_Label'] = predicted_labels_gb

# Save the DataFrame with the predicted labels
first_training.to_csv('labeled_first_training_gb.csv', index=False)

feature_importances_gb = gb_classifier.feature_importances_
feature_names = X_train.columns
importances_gb = pd.Series(feature_importances_gb, index=feature_names)
sorted_importances_gb = importances_gb.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_importances_gb, y=sorted_importances_gb.index)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance for Gradient Boosting')
plt.show()
