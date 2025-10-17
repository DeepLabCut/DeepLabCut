import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results CSV file
results_file = 'all_results.csv'
results_df = pd.read_csv(results_file)

# Group by experiment
grouped = results_df.groupby('experiment')

# Prepare data for plotting
data_all = [group['all__test rmse'].values for _, group in grouped]
labels = [experiment for experiment, _ in grouped]

# Create boxplot
fig, ax = plt.subplots(figsize=(4, 3))
ax.boxplot(data_all, labels=labels, widths=0.6)
ax.set_ylabel('all__test rmse')
ax.set_xlabel('Experiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('all__test_rmse_by_experiment.png', dpi=300)
plt.close()
# plt.show()

# Calculate RMSE deviation from control for each shuffle
deviations = []

for shuffle_num, group in results_df.groupby('Shuffle number'):
    # Get control RMSE for this shuffle
    control_rmse_all = group[group['experiment'] == 'control']['all__test rmse'].values
    control_rmse_truncated = group[group['experiment'] == 'control']['truncated__test rmse'].values
    control_rmse_non_truncated = group[group['experiment'] == 'control']['non_truncated__test rmse'].values

    if len(control_rmse_all) == 0:
        print(f"Warning: No control found for shuffle {shuffle_num}")
        continue

    control_rmse_all = control_rmse_all[0]
    control_rmse_truncated = control_rmse_truncated[0]
    control_rmse_non_truncated = control_rmse_non_truncated[0]

    # Calculate deviation for each experiment in this shuffle
    for experiment in group['experiment'].unique():
        if experiment == 'control':
            continue

        exp_rmse_all = group[group['experiment'] == experiment]['all__test rmse'].values
        exp_rmse_truncated = group[group['experiment'] == experiment]['truncated__test rmse'].values
        exp_rmse_non_truncated = group[group['experiment'] == experiment]['non_truncated__test rmse'].values
        if len(exp_rmse_all) > 0:
            deviation_all = exp_rmse_all[0] - control_rmse_all
            deviation_truncated = exp_rmse_truncated[0] - control_rmse_truncated
            deviation_non_truncated = exp_rmse_non_truncated[0] - control_rmse_non_truncated
            deviations.append({
                'experiment': experiment,
                'shuffle': shuffle_num,
                'rmse_deviation_all': deviation_all,
                'rmse_deviation_truncated': deviation_truncated,
                'rmse_deviation_non_truncated': deviation_non_truncated,
                'relative_rmse_deviation_all': deviation_all / control_rmse_all,
                'relative_rmse_deviation_truncated': deviation_truncated / control_rmse_truncated,
                'relative_rmse_deviation_non_truncated': deviation_non_truncated / control_rmse_non_truncated,
            })

# Create dataframe with deviations
deviations_df = pd.DataFrame(deviations)

# Group by experiment and plot box and whisker plots
fig, ax = plt.subplots(figsize=(4, 3))

# Prepare data for boxplot
experiments = deviations_df['experiment'].unique()
data_all = [deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_all'].values
        for exp in experiments]
data_truncated = [deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_truncated'].values
        for exp in experiments]
data_non_truncated = [deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_non_truncated'].values
        for exp in experiments]
means_all = [(exp, np.mean(deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_all'].values))
        for exp in experiments]
relative_means_all = [(exp, np.mean(deviations_df[deviations_df['experiment'] == exp]['relative_rmse_deviation_all'].values))
        for exp in experiments]

means_truncated = [(exp, np.mean(deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_truncated'].values))
        for exp in experiments]
means_non_truncated = [(exp, np.mean(deviations_df[deviations_df['experiment'] == exp]['rmse_deviation_non_truncated'].values))
        for exp in experiments]

print(means_all)
print(relative_means_all)

data = data_all + data_truncated + data_non_truncated
labels = [f'{exp}_all' for exp in experiments] + [f'{exp}_t' for exp in experiments] + [f'{exp}_nt' for exp in experiments]

ax.boxplot(data, labels=labels, widths=0.6)
ax.set_ylabel(r'RMSE Deviation from Control')
ax.set_xlabel('Landmark Subset')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Control baseline')
# ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# plt.show(block=True)
plt.savefig('rmse_deviation_by_experiment.png', dpi=300)
# plt.show()



