# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and Preprocess Data
# Load the dataset
file_path = 'results.csv'
data = pd.read_csv(file_path)

# Handle missing values if any
data = data.dropna()

# Display the first few rows of the dataset
data.head()

# Update column names to match the dataset
# Correct column renaming to match the actual dataset
# Original columns: ['est', 'err', 'est_card', 'true_card', 'query_dur_ms']
data.columns = ['est', 'err', 'est_card', 'true_card', 'query_dur_ms']

total_size = 12303280

# Filter Low Selectivity Queries
data = data[data['true_card'] / total_size < 0.005]

# Compute Statistical Metrics from 'err' column
median_error = data['err'].median()
percentile_95_error = np.percentile(data['err'], 95)
percentile_99_error = np.percentile(data['err'], 99)
max_error = data['err'].max()

# Print the results
print(f"Median Error: {median_error}")
print(f"95th Percentile Error: {percentile_95_error}")
print(f"99th Percentile Error: {percentile_99_error}")
print(f"Max Error: {max_error}")