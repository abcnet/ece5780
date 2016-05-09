import subprocess
import os
import os.path

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


import matplotlib.pyplot as plt # slow
patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]
done = 9
es = [[16, 17, 18], [10, 11, 12], [11, 12, 13], [13, 14, 15], [11, 12, 13], [13, 14, 15], [13, 14, 15], [10, 11, 12], [10, 11, 12]]

findPath = '../!Data/train/'
saxList = [[] for _ in range(done)]


for i in range(done):
	patient = patients[i]
	print 'Patient', patient
	path = findPath + str(patient) + '/study/'
	cmd = "ls -d " + path + 'sax_*'
	saxList[i] = map(lambda x: int(x[x.find('_')+1:]), subprocess.check_output(cmd, shell=True).split())
	saxList[i].sort()
	# print saxList[i]
	for sax in saxList[i]:
		saxPath = path + 'sax_' + str(sax)
		cmd = 'ls "' + saxPath + '"'
		imgs = subprocess.check_output(cmd, shell=True).split()
		# print imgs
		for imgName in imgs:
			imgPath = saxPath + '/' + imgName
			# print imgPath
			i1 = imgName.find('-00')
			i2 = imgName.find('.dcm')
			timeFrame = imgName[i1+3:i2]
			do = 0
			if int(timeFrame) in es[i]:
				do = int(timeFrame)
				# print timeFrame, imgPath
				
			if int(timeFrame) == 1:
				do = 1
				# print timeFrame, imgPath
			if do>0:
				segmentationPath = 'tiftest/' + str(patient) + '/sax_' + str(sax) + '.1-single-' + str(do) + '.png'
				# print segmentationPath
				if os.path.isfile(segmentationPath):
					print timeFrame, imgPath, segmentationPath
					segmentation = imread(segmentationPath)
					# print segmentation
					segmentationtorgb = cv2.cvtColor(np.array(segmentation, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
					plt.imshow(segmentationtorgb)
					plt.show()

# print saxList