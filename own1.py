import cv2
import numpy as np
import dicom
import json
import os
import random
import re
import shutil
import sys

# from matplotlib import image # slow

from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline



import matplotlib.pyplot as plt # slow
# import matplotlib.image as mpimg # slow

import scipy
# import pylab # slow
import numpy as np
from segment import Dataset
from scipy.misc import imsave, imread



patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]


path1 = '/Users/zr/Downloads/!ECE5780/!Kaggle/!Data/train/28/study/sax_5'
img1 = path1 + '/' + 'IM-14207-0001.dcm'

if __name__ == "__main__":
    ds = dicom.read_file(img1)
    # pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    # pylab.show()

    img = np.array(ds.pixel_array)
    imsave('raw.png', img)
    img8 = imread('raw.png')


    # plt.imshow(img)
    # plt.show()

    # colorimg = np.array([img,img,img], dtype=np.uint8)
    # cv2.circle(colorimg,(91,139), 5, (255,255,0), -1)


    # import matplotlib.pyplot as plt
    # img8 = np.array(img, dtype=np.uint8)

    backtorgb = cv2.cvtColor(img8,cv2.COLOR_GRAY2RGB)

    cv2.circle(backtorgb,(91,120), 2, (255,255,0), -1)

    # cmap = plt.get_cmap('jet')

    # rgba_img = cmap(img)
    # rgb_img = np.delete(rgba_img, 3, 2)
    plt.imshow(backtorgb)
    plt.show()

    imsave('test.png', backtorgb)
