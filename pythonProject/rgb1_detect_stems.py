import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv
from skimage import img_as_float
from skimage.transform import probabilistic_hough_line

########################################################################################################################
# Parameters
########################################################################################################################
img_path = 'C:/Huajian/project2021/crown_rot_0590/images/check stems'
# img_name = 'IMG20210714152350.jpg'
img_name = 'IMG20210714152818.jpg'

t_blue = 0.09 # Threshold of blue image
t_saturation = 0.7

hough_threshold = 50
hough_line_length = 100
hough_line_gap = 10

########################################################################################################################
# RGB image
########################################################################################################################
# Read image and convert to double in [0, 1]
rgb_uint8 = plt.imread(img_path + '/' + img_name)
rgb = img_as_float(rgb_uint8)

f1, ax_f1 = plt.subplots(1, 2)
ax_f1[0].imshow(rgb_uint8)
ax_f1[0].set_title('RGB image in unit8 format')
ax_f1[1].imshow(rgb, cmap='jet')
ax_f1[1].set_title('RGB image in float64 format')
plt.pause(0.01)

# R, G and B channels
red = rgb[0:100, :, 0]
green = rgb[:, :, 1]
blue = rgb[:, :, 2]

f2, ax_f2 = plt.subplots(2, 2)
ax_f2[0, 0].imshow(rgb)
ax_f2[0, 0].set_title('RGB image')

ax_f2[0, 1].imshow(red, cmap='gray')
ax_f2[0, 1].set_title('red')

ax_f2[1, 0].imshow(green, cmap='jet')
ax_f2[1, 0].set_title('green')

ax_f2[1, 1].imshow(blue, cmap='jet')
ax_f2[1, 1].set_title('blue')
plt.pause(0.01)


########################################################################################################################
# Hue, saturation and values
########################################################################################################################
hsv = rgb2hsv(rgb)
hue = hsv[:, :, 0]
saturation = hsv[:, :, 1]
value = hsv[:, :, 2]

f3, ax_f3 = plt.subplots(2, 2)
ax_f3[0, 0].imshow(hsv, cmap='jet')
ax_f3[0, 0].set_title('HSV image')

ax_f3[0, 1].imshow(hue, cmap='jet')
ax_f3[0, 1].set_title('Hue')

ax_f3[1, 0].imshow(saturation, cmap='jet')
ax_f3[1, 0].set_title('Saturation')

ax_f3[1, 1].imshow(value, cmap='jet')
ax_f3[1, 1].set_title('value')
plt.pause(0.01)

########################################################################################################################
#
########################################################################################################################
bw = np.logical_and(blue < t_blue, saturation > t_saturation)

f4, ax_f4 = plt.subplots(1, 2)
ax_f4[0].imshow(bw, cmap='gray')
plt.pause(0.01)

lines = probabilistic_hough_line(bw, threshold=hough_threshold,
                                 line_length=hough_line_length,
                                 line_gap=hough_line_gap)

ax_f4[1].imshow(rgb)

for i in range(0, 5):
    p0, p1 = lines[i]
    ax_f4[1].plot((p0[0], p1[0]), (p0[1], p1[1]), color='red')
ax_f4[1].set_xlim((0, rgb.shape[1]))
ax_f4[1].set_ylim((rgb.shape[0], 0))
ax_f4[1].set_title('Probabilistic Hough')


