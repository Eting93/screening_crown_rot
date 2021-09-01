from plantcv import plantcv as pcv
import matplotlib
from skimage.transform import rotate
from matplotlib import pyplot as plt

class options:
    def __init__(self):
        self.image = "D:\crown_rot_image\colour\second\H\infected\First\cut086762.jpg"
        self.debug = "plot"
        self.writeimg = False
        self.result = "vis_tutorial_results.json"
        self.outdir = "."  # Store the output to the current directory


# Get options
args = options()

# Set debug to the global parameter
pcv.params.debug = args.debug
img, path, filename = pcv.readimage(filename=args.image)
s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
color_histogram = pcv.analyze_color(rgb_img=img, mask=None, hist_plot_type=None, label="default")
pcv.print_image(img=color_histogram, filename="vis_tutorial_color_hist.jpg")
