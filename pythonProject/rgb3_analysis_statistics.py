import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import numpy as np

# Parameters
data_path = 'C:/Huajian/python_projects/crow_rot_0590/crown_rot_0590_processed_data'
data_name = 'rgb_scores_statistics.xlsx'

# Load data
sco_sta = pd.read_excel(data_path + '/' + data_name)

# The pearson correlation between tha average scores and the statistics
record_corr = {}
for i in range(13, 13 + 9):
    corr, _ = pearsonr(sco_sta['Average(28_07)'], sco_sta.iloc[:, i])
    record_corr[sco_sta.columns[i]] = corr

record_corr = pd.DataFrame.from_dict(record_corr, orient='index')
print(record_corr)

# Plot the pearson correlation coefficient
fig, ax = plt.subplots()
bar = ax.bar(record_corr.index, np.abs(record_corr.iloc[:, 0]))
ax.set_ylabel('Pearson correlation coefficient (abs)')
ax.grid(True, axis='y', linestyle='dashed')



