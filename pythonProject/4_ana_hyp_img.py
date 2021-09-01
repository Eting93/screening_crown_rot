import numpy as np
from matplotlib import pyplot as plt
from sklearn .decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from project_tools import *

########################################################################################################################
# Parameters
########################################################################################################################
pro_data_path = 'C:/Huajian/python_projects/crow_rot_0590/crown_rot_0590_processed_data'
# data_name = 'processed_hypimg_20210506.npy'
# data_name = 'processed_hypimg_20210520.npy'
data_name = 'processed_hypimg_20210526.npy'

# input_data_type = 'pcs' # pcs, ref
input_data_type = 'ref'

num_classes = 2

# classification_method = 'k_means'
# classification_method = 'som'
classification_method = 'nn'

n_comp_pca = 10

# K-means parameters
km_random_state = 42

# SOM parameters:
som_map_row = 20
som_map_col = 20
som_flag_pbc = True

# nn parameters
nn_num_hidden = 64
nn_batch_size = 3
nn_epoch = 20


########################################################################################################################
# Load data
########################################################################################################################
data = np.load(pro_data_path + '/' + data_name, allow_pickle=True)
data = data.flat[0]


########################################################################################################################
# Remove "No hyper-data" and "all-zeros"
########################################################################################################################
ref_data = []
barcode = []
variety = []
treatment1 = []
vnir_name = []
for i in range(0, data['ave_ref'].shape[0]):
    # if data[i] is 'No hyper-data' or all-zeros
    if data['ave_ref'][i][0] == 'N' or (data['ave_ref'][i] == 0).all():
        pass
    else:
        ref_data.append(data['ave_ref'][i])
        barcode.append(data['barcode'][i])
        variety.append(data['variety'][i])
        treatment1.append(data['treatment1'][i])
        vnir_name.append(data['vnir'][i])

ref_data = np.asarray(ref_data)
barcode = np.asarray(barcode)
variety = np.asarray(variety)
treatment1 = np.asarray(treatment1)
vnir_name = np.asarray(vnir_name)

# Separate controlled and infected
ref_con = ref_data[treatment1 == 'control']
ref_inf = ref_data[np.logical_not(treatment1 == 'control')]
lab_num_two_cla = np.ones((ref_data.shape[0], ))
lab_num_two_cla[treatment1 == 'control'] = 0
lab_all = {'string': treatment1, 'num_two_cla': lab_num_two_cla}

########################################################################################################################
# Plot data
########################################################################################################################
plt.figure()
for i in range(0, ref_con.shape[0]):
    plt.plot(data['wavelengths'], ref_con[i], color='green')

for i in range(0, ref_inf.shape[0]):
    plt.plot(data['wavelengths'], ref_inf[i], color='red', linestyle='dashed')

plt.plot(0, 0, color='green', label='Controlled')
plt.plot(0, 0, color='red', linestyle='dashed', label='Infected')
plt.legend()

########################################################################################################################
# Input data type
########################################################################################################################
if input_data_type == 'ref':
    input_data = ref_data
elif input_data_type == 'pcs':
    pca = PCA(n_components=n_comp_pca)
    pcs = pca.fit_transform(ref_data)
    pcs_con = pcs[0:ref_con.shape[0], :]
    pcs_inf = pcs[ref_con.shape[0]:, :]

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

    input_data = pcs
else:
    print('Wrong input data of SOM')


########################################################################################################################
# Classification
########################################################################################################################
# Organise training data and testing data
input_data_train = input_data
input_data_test = input_data
label_train = lab_all
label_test = lab_all

# Train
if classification_method == 'som':
    som(som_map_row, som_map_col, input_data_train, labels=lab_all, flag_pbc=som_flag_pbc)
elif classification_method == 'k_means':
    k_means(input_data, km_random_state, num_classes, lab_all)
elif classification_method == 'nn':

    nn_multiclass_logistic_regression(input_data, label_train['num_two_cla'], num_hidden=nn_num_hidden, num_output=num_classes,
                                      batch_size=nn_batch_size, learning_rate=0.01, epochs=nn_epoch,
                                      smoothing_constant=0.01, flag_shuffle=True)
else:
    print('Wrong classification method!')

# Test



