from plantcv import plantcv as pcv
class options:
    def __init__(self):
        self.image = "D:\crown_rot_image\colour\pythoncut/086622.jpg"
        self.debug = "plot"
        self.writeimg = False
        self.result = "vis_tutorial_results.json"
        self.outdir = "."  # Store the output to the current directory


# Get options
args = options()

# Set debug to the global parameter
pcv.params.debug = args.debug

# Get the image
img, path, filename = pcv.readimage(filename=args.image)  # figure 1

# Get the HSV result
s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')  # figure 2
# first threshold based on saturation (120)
s_thresh = pcv.threshold.binary(gray_img=s, threshold=120, max_value=255, object_type='light')  # figure 3
# Median Blur for cleaning the noise
s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5) # figure 4

# Get the LAB result
b = pcv.rgb2gray_lab(rgb_img=img, channel='b')  # figure 5
# Second threshold based on "blue-yellow" (155)
b_thresh = pcv.threshold.binary(gray_img=b, threshold=155, max_value=255,
                                object_type='light')  # figure 6
# Combining (1) the both threshold b and s
combination1 = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)  # figure 7
# first mask for the combination 'b' and 's'
masked = pcv.apply_mask(img=img, mask=combination1, mask_color='white')  # figure 8
# extract the 'Green-Magenta' and 'Blue-Yellow' channels
masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')  # figure 9
masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')  # figure 10

# threshold for the 'a' = 'Green-Magenta'
maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=125,
                                      max_value=255, object_type='dark')  # figure 11
# threshold for the 'a' = 'Green-Magenta'
maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=130,
                                       max_value=255, object_type='light')  # figure 12
# threshold for the 'b' = 'Blue-Yellow' (same to fig 6)
maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=155,
                                      max_value=255, object_type='light')  # figure 13
# combination 2
ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)  # figure 14
ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)  # figure 15

opened_ab = pcv.opening(gray_img=ab)   # figure 16
# Fill small objects (reduce image noise) requiring 200 to 3500
ab_fill = pcv.fill(bin_img=ab, size=3000)  # figure 17
closed_ab = pcv.closing(gray_img=ab_fill)  # figure 18
# applying second mask
masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')  # figure 19
# Identify objects
id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)  # figure 20
roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=400, y=2000, h=300, w=800)  # figure 21
# keep the objects
roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1,
                                                               roi_hierarchy=roi_hierarchy,
                                                               object_contour=id_objects,
                                                               obj_hierarchy=obj_hierarchy,
                                                               roi_type='partial')  # figure 22
obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy3)  # figure 23

# analysing the imaging the colour
color_histogram = pcv.analyze_color(rgb_img=img, mask=kept_mask, colorspaces='all', label="default")  # figure 24
pcv.print_image(img=color_histogram, filename="crown_rot_color_hist.jpg")
top_x, bottom_x, center_v_x = pcv.x_axis_pseudolandmarks(img=img, obj=obj, mask=mask, label="default")
top_y, bottom_y, center_v_y = pcv.y_axis_pseudolandmarks(img=img, obj=obj, mask=mask)

