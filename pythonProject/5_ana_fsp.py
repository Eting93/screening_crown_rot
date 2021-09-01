import pandas as pd
from matplotlib import pyplot as plt
from sklearn .decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import SimpSOM as sps

# Parameters
data_path = 'C:\Huajian\python_projects\crow_rot_0590\crown_rot_0590_processed_data'
# data_name = 'crown_rot_fsp_20210504.xlsx'
data_name = 'crown_rot_fsp_20210512.xlsx'
input_data_som = 'pcs'

# Read data
data_frame = pd.read_excel(data_path + '/' + data_name, sheet_name='raw_data')
wave = data_frame.columns[1:]

ref_con = []
ref_inf = []
lab_con = []
lab_inf = []
for a_row in data_frame.iterrows():
    if a_row[1]['Wavelength'][0:7] == 'control' or a_row[1]['Wavelength'][0:7] == 'Control':
        ref_con.append(a_row[1].values[1:])
        lab_con.append('C')
    else:
        ref_inf.append(a_row[1].values[1:])
        lab_con.append('I')

ref_con = np.asarray(ref_con)
ref_inf = np.asarray(ref_inf)
lab_con = np.asarray(lab_con)
lab_inf = np.asarray(lab_inf)

# Plot control
plt.figure()
for i in range(0, ref_con.shape[0]):
    plt.plot(wave, ref_con[i], color='green')
plt.plot(0, 0, color='green', label='Controlled')

# Plot infected
for i in range(0, ref_inf.shape[0]):
    plt.plot(wave, ref_inf[i], color='red', linestyle='dashed')
plt.plot(0, 0, color='red', linestyle='dashed', label='Infected')
plt.xlabel('Wavelength (nm)', fontsize='14', fontweight='bold')
plt.ylabel('Reflectance', fontsize='14', fontweight='bold')
plt.legend()
plt.show()

# PCA
ref_all = np.concatenate((ref_con, ref_inf), axis=0)
pca = PCA(n_components=3)
pcs = pca.fit_transform(ref_all)
pcs_con = pcs[0:ref_con.shape[0], :]
pcs_inf = pcs[ref_con.shape[0]: ]

# 2D plot of pc1 and pc2
plt.figure()
plt.scatter(pcs_con[:, 0], pcs_con[:, 1], c='green', label='Controlled')
plt.scatter(pcs_inf[:, 0], pcs_inf[:, 1], c='red', label='Infected')
plt.legend()
plt.xlabel('PC1', fontweight='bold')
plt.ylabel('PC2', fontweight='bold')

# 3D plot of pc1, pc2 and pc3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcs_con[:, 0], pcs_con[:, 1], pcs_con[:, 2], c='green', label='Controlled')
ax.scatter(pcs_inf[:, 0], pcs_inf[:, 1], pcs_inf[:, 2], c='red', label='Infected')
plt.legend()
ax.set_xlabel('PC1', fontweight='bold')
ax.set_ylabel('PC2', fontweight='bold')
ax.set_zlabel('PC3', fontweight='bold')
plt.pause(1)


# ########################################################################################################################
# # SOM
# ########################################################################################################################
print('SOM')

# 1. load data
if input_data_som == 'pcs':
    input_data = pcs
elif input_data_som == 'ref':
    input_data = ref_all
else:
    print('Wrong input data of SOM')

lab_all = np.concatenate((lab_con, lab_inf), axis=0)

# Build a network 20x20 with a weights format taken from the ref and activate Periodic Boundary Conditions.
net = sps.somNet(20, 20, input_data, PBC=True)

# Train the network for 10000 epochs and with initial learning rate of 0.01.
net.train(0.01, 1000)

# Save the weights to file
net.save('som_weights')
weight = np.load('som_weights.npy') # (401, 2151)

net.nodes_graph(colnum=0) # Plot a figure of node feature (column 0) and save the figure in the PWD
net.diff_graph() # Plot a figure of weight difference and save it in the PWD

# Project the datapoints on the new 2D network map.
net.project(input_data, labels=lab_all) # Project the labels to the weight different figure and save it.

# Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(input_data, type='qthresh') # Do clustering and save a figure




