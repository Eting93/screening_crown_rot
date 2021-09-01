from project_tools import rgb_stem_statistics
from os import walk
import pandas as pd


########################################################################################################################
# Parameters
########################################################################################################################
data_path = 'D:\crown_rot_image\colour\pythoncut\cut_20_07'
flag_figure = False
flag_save = True

score_data_path = 'D:\crown_rot_image\colour\coding\pythonProject'
score_data_name = 'munal_cut_28_07_result.xlsx'
sheet_name = 'score'


########################################################################################################################
# Read image names and the file of scores
########################################################################################################################
# Organise image path and names
for (root, dirs, image_names) in walk(data_path):
    break

# Read scores
scores = pd.read_excel(score_data_path + '/' + score_data_name, sheet_name=sheet_name)

# Make news columns for the statistics
scores['ave_red'] = 'none'
scores['ave_green'] = 'none'
scores['ave_blue'] = 'none'
scores['ave_h'] = 'none'
scores['ave_s'] = 'none'
scores['ave_v'] = 'none'
scores['ave_l'] = 'none'
scores['ave_a'] = 'none'
scores['ave_b'] = 'none'

for i in range(0, 10):
    scores['hist_h' + str(i)] = 'none'

for i in range(0, 10):
    scores['hist_a' + str(i)] = 'none'


########################################################################################################################
# Calculate statistics
########################################################################################################################
scores_statistics = []
for a_image_name in image_names:
    stem_sta = rgb_stem_statistics(data_path, a_image_name, flag_figure)

    for i in range(0, scores.shape[0]):
        if str(scores['Serial No.'][i]) == a_image_name[4:-4]:
            # Writ the statistics to the excel file.
            scores.loc[i, 'ave_red'] = stem_sta['average red']
            scores.loc[i, 'ave_green'] = stem_sta['average green']
            scores.loc[i, 'ave_blue'] = stem_sta['average blue']
            scores.loc[i, 'ave_h'] = stem_sta['average hue']
            scores.loc[i, 'ave_s'] = stem_sta['average saturation']
            scores.loc[i, 'ave_v'] = stem_sta['average value']
            scores.loc[i, 'ave_l'] = stem_sta['average lightness']
            scores.loc[i, 'ave_a'] = stem_sta['average a']
            scores.loc[i, 'ave_b'] = stem_sta['average b']

            for j in range(0, 10):
                scores.loc[i, 'hist_h' + str(j)] = stem_sta['hist_h'][j]
                scores.loc[i, 'hist_a' + str(j)] = stem_sta['hist_a'][j]

            # Record the statistics to scores_statistics
            scores_statistics.append(scores.iloc[i])

    print(a_image_name + ' finished.')

########################################################################################################################
# Save
########################################################################################################################
if flag_save:
    writer = pd.ExcelWriter('rgb_scores_statistics.xlsx', engine='xlsxwriter')
    pd.DataFrame(scores_statistics).to_excel(writer, sheet_name='sco_sta_clear')
    scores.to_excel(writer, sheet_name='sco_sta')
    writer.save()
    print('Data saved!')





