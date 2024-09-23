import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the merged dataset
data = pd.read_csv('merged_training_dataset.csv')

features_data = data.drop(data.columns[0], axis=1)

# Separate features and labels
X = features_data.iloc[:, :-1].values  # The last column is the label
y = features_data.iloc[:, -1].values

# Split the dataset into training and testing sets (80/20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(x_train, y_train)

# Predict on the testing set
y_pred = model.predict(x_test)

# Evaluate the model
print(f"Accuracy on Test Set: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))



# Save the model to disk
filename = 'finalized_model20.sav'
joblib.dump(model, filename)


# Load the model from disk
loaded_model = joblib.load(filename)
