import pandas as pd
from scipy.stats import shapiro, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scikit_posthocs as sp

# Load the data from a CSV file with the correct delimiter
data = pd.read_csv('Project8DataCSV.csv', delimiter=';')

# Clean column names
data.columns = data.columns.str.strip()

# Define the attributes
attributes = [
    'PositivExperience', 'Engagement', 'Education', 'Fascination', 'Autenticity',
    'Distracting', 'Irritating', 'Clearly', 'Annoying', 'Understandable',
    'Disturbing', 'Realistic', 'Pleasant', 'Satisfaction'
]

# Convert relevant columns to numeric
for column in attributes:
    data[column] = pd.to_numeric(data[column], errors='coerce')

# Calculate descriptive statistics
summary_stats = data.groupby('Condition')[attributes].agg(['mean', 'std']).reset_index()
print(summary_stats)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a box plot for each attribute
for attribute in attributes:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Condition', y=attribute, data=data)
    plt.title(f'Box Plot of {attribute} by Condition')
    plt.show()

# Create a bar plot for mean values of each attribute
mean_values = data.groupby('Condition')[attributes].mean().reset_index()
mean_values_melted = mean_values.melt(id_vars='Condition', value_vars=attributes, var_name='Attribute', value_name='Mean')

plt.figure(figsize=(12, 8))
sns.barplot(x='Attribute', y='Mean', hue='Condition', data=mean_values_melted)
plt.title('Mean Values of Attributes by Condition')
plt.xticks(rotation=70, ha='right')  # Rotate labels and align them to the right
plt.tight_layout()  # Adjust the padding to fit labels
plt.savefig('mean_values_barplot.pdf')  # Save as PDF
plt.show()

# Check for normal distribution
normality_results = {}
for attribute in attributes:
    normality_results[attribute] = {}
    for condition in data['Condition'].unique():
        stat, p_value = shapiro(data[data['Condition'] == condition][attribute].dropna())
        normality_results[attribute][condition] = p_value
        print(f"Shapiro-Wilk test for {attribute} under {condition}: p-value={p_value}")

# Determine if data is normally distributed for each attribute
normal_distribution = {attribute: all(p > 0.05 for p in normality_results[attribute].values()) for attribute in attributes}
print(normal_distribution)

# Perform ANOVA or Kruskal-Wallis based on normality
anova_results = {}
kruskal_results = {}
for attribute in attributes:
    groups = [data[data['Condition'] == condition][attribute].dropna() for condition in data['Condition'].unique()]
    if normal_distribution[attribute]:
        # Perform ANOVA
        stat, p_value = f_oneway(*groups)
        anova_results[attribute] = (stat, p_value)
        print(f"ANOVA for {attribute}: F={stat}, p={p_value}")
    else:
        # Perform Kruskal-Wallis
        stat, p_value = kruskal(*groups)
        kruskal_results[attribute] = (stat, p_value)
        print(f"Kruskal-Wallis for {attribute}: H={stat}, p={p_value}")

# Conduct post hoc tests if Kruskal-Wallis is significant
significant_attributes = [attribute for attribute in kruskal_results if kruskal_results[attribute][1] < 0.05]

for attribute in significant_attributes:
    print(f"\nPost Hoc Tests for {attribute}")

    # Prepare data for Dunn test
    data_subset = data[['Condition', attribute]].dropna()
    conditions = data['Condition'].unique()

    # Perform Dunn test
    dunn_results = sp.posthoc_dunn(data_subset, val_col=attribute, group_col='Condition', p_adjust='bonferroni')

    # Print Dunn test results
    for condition1, condition2 in itertools.combinations(conditions, 2):
        p_value = dunn_results.loc[condition1, condition2]
        print(f"{condition1} vs {condition2}: p={p_value}")

