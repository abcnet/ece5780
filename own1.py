import cv2
import numpy as np
import dicom
import json
import os
import random
import re
import shutil
import sys
from matplotlib import image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import pylab
import numpy as np
from segment import Dataset

patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]


path1 = '/Users/zr/Downloads/!ECE5780/!Kaggle/!Data/train/28/study/sax_5'
img1 = path1 + '/' + 'IM-14207-0001.dcm'

if __name__ == "__main__":
    ds = dicom.read_file(img1)
    pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    # imgarray = np.array(ds.pixel_array)
    pylab.show()