import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

keep_cols = [
    'experiment',
#     'Shuffle number',
#     'fold',
    'seed',
    'all__test rmse',
    'truncated__test rmse',
    'non_truncated__test rmse',
]

# Load the results CSV file
results_files = [
    'results_control_4x4.csv',
    'results_ll-0d1_4x4.csv',
    'results_xray-1d15_4x4.csv',
]
result_dfs = [pd.read_csv(f)[keep_cols] for f in results_files]

results_df = pd.concat(result_dfs)

# Group by experiment and seed, and average the RMSEs
grouped = results_df.groupby(['experiment', 'seed']).mean(numeric_only=True).reset_index()

# print(grouped)

grouped_2 = grouped.groupby('experiment').aggregate({
    'all__test rmse': ['mean', 'std', 'min'],
    'truncated__test rmse': ['mean', 'std', 'min'],
    'non_truncated__test rmse': ['mean', 'std', 'min'],
})

print(grouped_2)
