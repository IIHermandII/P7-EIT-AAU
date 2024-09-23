import pandas as pd

# Load the two datasets
df1 = pd.read_csv('labeled_first_training_rf_imp.csv')
df2 = pd.read_csv('labeled_second_training_rf.csv')

# Concatenate the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged dataset to a new CSV file
merged_df.to_csv('merged_training_dataset.csv', index=False)
