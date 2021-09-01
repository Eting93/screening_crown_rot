import pandas as pd
import sys
sys.path.append('C:/Huajian/python_projects/appf_toolbox_project')
from appf_toolbox.machine_learning import classification as cl
from sklearn .decomposition import PCA
from matplotlib import pyplot as plt

########################################################################################################################
# Parameters
########################################################################################################################
data_path = 'D:/crown_rot_image/colour/coding/pythonProject'
data_name = 'rgb_scores_statistics_manual_addnew_ANOVA _remove_Aurora.xlsx'

rkf_n_splits = 5
rkf_n_repeats = 3
rkf_random_state = 1

opt_num_iter_cv = 5
opt_num_fold_cv = 5
opt_num_evals = 3

svm_kernel = 'rbf'
svm_c_range = [1, 100]
svm_gamma_range = [1, 50]
svm_tol = 1e-3

original_data_type = 'HIST_a'
transformation = 'PCA'

n_comp_pca = 3

flag_save = False


########################################################################################################################
# Training data and testing data
########################################################################################################################
# Read the data from the excel file
data_sheet = pd.read_excel(data_path + '/' + data_name, sheet_name='sco_sta_clear')

# Read input data and label of classification model
if original_data_type == 'HIST_a':
    ori_data = data_sheet.loc[:, 'hist_h0':'hist_a8']
# elif:
    # Other data type here

label = data_sheet['Disease']

# Convert data to numpy array
ori_data = ori_data.to_numpy()
label = label.to_numpy()

# Change label to 1 for infected and 0 for control
label[label=='Infected'] = 1
label[label=='Control'] = 0
label = label.astype(int)


########################################################################################################################
# Transformation
########################################################################################################################
input_type = original_data_type
if transformation == 'PCA':
    input_type = original_data_type + '_' + transformation
    pca = PCA(n_components=n_comp_pca)
    pcs = pca.fit_transform(ori_data)

    # 2D plot of pc1 and pc2
    plt.figure()
    plt.title('PC 1 and 2')
    for i in range(0, pcs.shape[0]):
        if label[i] == 0:
            plt.scatter(pcs[i, 0], pcs[i, 1], c='green')
        else:
            plt.scatter(pcs[i, 0], pcs[i, 1], c='red')
    plt.scatter(0, 0, c='green', label='Controlled')
    plt.scatter(0, 0, c='red', label='Infected')
    plt.legend()
    plt.xlabel('PC1', fontweight='bold')
    plt.ylabel('PC2', fontweight='bold')

    # 3D plot of pc1, pc2 and pc3
    fig = plt.figure()
    fig.suptitle('PC 1, 2 and 3')
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0, pcs.shape[0]):
        if label[i] == 0:
            ax.scatter(pcs[i, 0], pcs[i, 1], pcs[i, 2], c='green')
        else:
            ax.scatter(pcs[i, 0], pcs[i, 1], pcs[i, 2], c='red')
    ax.scatter(0, 0, 0, c='green', label='Controlled')
    ax.scatter(0, 0, 0, c='red', label='Infected')
    plt.legend()
    ax.set_xlabel('PC1', fontweight='bold')
    ax.set_ylabel('PC2', fontweight='bold')
    ax.set_zlabel('PC3', fontweight='bold')
    plt.pause(30)

    input = pcs


########################################################################################################################
# Check samples
########################################################################################################################
cl.plot_samples_with_colourbar(input, label, input_type=input_type, title=input_type)

karg_tune_model = {'svm_kernel': svm_kernel,
                   'svm_c_range': svm_c_range,
                   'svm_gamma_range': svm_gamma_range,
                   'svm_tol': svm_tol,
                   'opt_num_iter_cv': opt_num_iter_cv,
                   'opt_num_fold_cv': opt_num_fold_cv,
                   'opt_num_evals': opt_num_evals}


########################################################################################################################
# Cross validation
########################################################################################################################
report_cv = cl.repeadted_kfold_cv(input, label,
                                  n_splits=rkf_n_splits,
                                  n_repeats=rkf_n_repeats,
                                  tune_model=cl.tune_svm_classification,
                                  karg=karg_tune_model,
                                  random_state=rkf_random_state,
                                  flag_save=flag_save)








