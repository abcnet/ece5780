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
seed_inner1 = (91,120)
seed_outer1 = (91,134)

class Label:
	def __init__(self, img, seed):
		self.img = img
		w, h = img.shape
		self.nrows = h
		self.ncols = w
		self.m = [-1] * h * w
		self.seed = seed
	# def setLabel(self, y, x, v):
	# 	self.m[y][x] = v + 1
	# def getLabel(self, y, x):
	# 	return self.m[y][x] - 1
	def yx2i(self, y, x):
		return y * self.ncols + x
	def i2yx(self, i):
		return i / self.ncols, i % self.ncols
	def find(self, i):
		# print 'find', i, self.m[i]
		# print 'self.m[0] = ', self.m[0]
		# print 'self.m[', i, '] = ', self.m[i]
		if self.m[i] < 0:
			# print 'After find, self.m[', i, '] < 0, self.m[0] = ', self.m[0]
			return i
		else:
			tmp = self.find(self.m[i])
			self.m[i] = tmp
			# print 'After find, self.m[', i, '] >= 0, self.m[0] = ', self.m[0]
			return tmp
	def union(self, i1, i2):
		# print 'union', i1, i2
		# print 'self.m[0] = ', self.m[0]
		root1 = self.find(i1)
		root2 = self.find(i2)
		# print 'union', root1, root2, 'instead'
		if root1 == root2:
			return
		v_root1 = self.m[root1]
		v_root2 = self.m[root2]
		if v_root2 < v_root1:
			self.m[root2] += v_root1
			self.m[root1] = root2
		else:
			self.m[root1] += v_root2
			self.m[root2] = root1
		# print 'After union ', i1, i2, ', self.m[0] = ', self.m[0]
	def getImg(self):
		# print 'size of img ', img.shape
		for y in range(self.nrows):
			for x in range(self.ncols):
				if x > 0 and self.img[x][y] == self.img[x-1][y]:
					self.union(self.yx2i(y, x - 1), self.yx2i(y, x))
				if y > 0 and self.img[x][y] == self.img[x][y-1]:
					self.union(self.yx2i(y - 1, x), self.yx2i(y, x))
		newimg = np.zeros(img.shape, dtype=np.uint8)
		# print 'size of newimg ', newimg.shape
		seedLabel = self.find(self.yx2i(self.seed[1], self.seed[0]))
		for y in range(self.nrows):
			for x in range(self.ncols):
				if self.find(self.yx2i(y, x)) == seedLabel:
					# print 'accessing ', x,y
					newimg[x][y] = 255
		return newimg






if __name__ == "__main__":
    ds = dicom.read_file(img1)
    # pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    # pylab.show()

    img = np.array(ds.pixel_array)

    # test thresholding
    
    # outer = img[seed_outer1]
    # inner = img[seed_inner1]
    # old_threshold = (img[seed_outer1] + img[seed_inner1]) / 2
    # print outer, inner, old_threshold, outer < old_threshold, inner > old_threshold, img[seed_outer1] < old_threshold, img[seed_inner1] > old_threshold


    inner = 0
    for x in (-1,0,1):
        for y in (-1,0,1):
     	   inner += img[seed_inner1[0]+x][seed_inner1[1]+y]
    inner /= 9

    threshold = (img[seed_outer1] + inner) * 4
    print(img[seed_outer1])
    print(inner)
    print(threshold)
    w,h = img.shape
    for x in range(w):
        for y in range(h):
            # print(x,y,img[x][y], threshold,  0 if img[x][y] < threshold else 65535)
            img[x][y] = 0 if img[x][y] < threshold else 65535
    labelMatrix = Label(img, seed_inner1)
    region = labelMatrix.getImg()


    imsave('region.png', region)

    imsave('16bit.png', img)   
    img8 = imread('16bit.png')
    region8 = imread('region.png')


    # plt.imshow(img)
    # plt.show()

    # colorimg = np.array([img,img,img], dtype=np.uint8)
    # cv2.circle(colorimg,(91,139), 5, (255,255,0), -1)


    # import matplotlib.pyplot as plt
    # img8 = np.array(img, dtype=np.uint8)

    backtorgb = cv2.cvtColor(img8,cv2.COLOR_GRAY2RGB)
    regiontorgb = cv2.cvtColor(region8,cv2.COLOR_GRAY2RGB)

    # for x in range(w):
    #     for y in range(h):
    #         print(x,y,img[x][y])

    cv2.circle(backtorgb,seed_inner1, 2, (255,255,0), -1)
    cv2.circle(backtorgb,seed_outer1, 1, (0,255,0), -1)

    cv2.circle(regiontorgb,seed_inner1, 2, (255,255,0), -1)
    cv2.circle(regiontorgb,seed_outer1, 1, (0,255,0), -1)

    # cmap = plt.get_cmap('jet')

    # rgba_img = cmap(img)
    # rgb_img = np.delete(rgba_img, 3, 2)
    plt.imshow(regiontorgb)
    plt.show()

    imsave('region2.png', regiontorgb)

