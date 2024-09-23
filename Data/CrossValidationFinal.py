import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('merged_training_dataset.csv')
features_data = data.drop(data.columns[0], axis=1)

# Apply LabelEncoder to the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(features_data.iloc[:, -1])  # Encode the last column

# Now, make sure to use all other columns as features
X = features_data.iloc[:, :-1].values

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform x-fold cross-validation
scores = cross_val_score(model, X, y, cv=7)

# Print the accuracy for each fold
print(f"Accuracy for each fold: {scores}")

# Print the mean accuracy and the 95% confidence interval of the score estimate
print(f"Mean accuracy: {scores.mean():.2f}")
print(f"95% confidence interval: {scores.mean() - 2 * scores.std():.2f} - {scores.mean() + 2 * scores.std():.2f}")

# Train the model on the entire dataset
model.fit(X, y)

