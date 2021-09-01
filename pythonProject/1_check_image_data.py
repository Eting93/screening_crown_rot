# Check if all project data was downloaded and if surplus data was downloaded according to
# processed_hypimg_data-time.xlsx.

import pandas as pd
from os import walk
import numpy as np

# Paramters
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210421'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210428'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210506'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210520'
hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210526'

processed_data_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data'
barcode_hypname_file = 'crown rot_barcode_hypname_20210526.xlsx'
sheet_name = 'barcode_hypname'


# Read barcode_hypname
barcode_hypname = pd.read_excel(processed_data_path + '/' + barcode_hypname_file, sheet_name=sheet_name)
hypnames_vnir = barcode_hypname['vnir']
hypnames_swir = barcode_hypname['swir']
hypnames = pd.concat((hypnames_vnir, hypnames_swir), axis=0)
hypnames = hypnames.to_list()

# Check if all merged data has been downloaded
print('Checking if all merged data in the sheet of ' + sheet_name + ' has been downloaded.')
for (root, dirs, files) in walk(hyp_data_path):
    hypnames_dl = dirs
    break

error_count = 0
nan_count = 0
for i in range(0, hypnames.__len__()):
    if hypnames[i] in hypnames_dl:
        pass
    elif pd.isnull(hypnames[i]):
        nan_count += 1
    else:
        error_count += 1
        print(hypnames[i] + ' was not downloaded')
print('A total of ' + str(error_count) + ' data was not downloaded.')
print('A total of ' + str(nan_count) + ' nan')

# Check if surplus data have been downloaded
print('Checking if surplus data was downloaded.')
error_count = 0
for i in range(0, hypnames_dl.__len__()):
    if hypnames_dl[i] in hypnames:
        pass
    else:
        error_count += 1
        print(hypnames_dl[i] + ' in is surplus data')
print('A total of ' + str(error_count) + ' data is surplus data')
        
