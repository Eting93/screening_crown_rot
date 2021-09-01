import sys
sys.path.append('/media/huajian/Files/python_projects/appf_toolbox_project')
from appf_toolbox.hyper_processing.pre_processing import green_plant_segmentation


# Parameters
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210421'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210428'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/20210506'
# hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210520'
hyp_data_path = '/media/huajian/Files/Data/crown_rot_pilot_0590_data/wiw_20210526'

model_name_vnir = 'green_crop_seg_model_OneClassSVM_vnir.npy'
model_name_swir = 'green_crop_seg_model_OneClassSVM_swir.npy'
model_path = '/media/huajian/Files/python_projects/p12_green_segmentation/green_seg_model_20210129'

# save_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/segmentation_20210421'
# save_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/segmentation_20210428'
# save_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/segmentation_20210520'
save_path = '/media/huajian/Files/python_projects/crown_rot_0590/crown_rot_0590_processed_data/segmentation_20210526'

green_plant_segmentation(hyp_data_path,
                        model_name_vnir=model_name_vnir,
                        model_name_swir=model_name_swir,
                        model_path=model_path,
                        gamma=0.8,
                        flag_remove_noise=True,
                        white_offset_top=0.1,
                        white_offset_bottom=0.9,
                        band_R=10,
                        band_G=50,
                        band_B=150,
                        flag_save=True,
                        save_path=save_path)