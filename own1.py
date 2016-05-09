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

TURN_ON_MEDIAN_FILTER = False

patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]


path1 = '/Users/zr/Downloads/!ECE5780/!Kaggle/!Data/train/28/study/sax_5'
img1 = path1 + '/' + 'IM-14207-0001.dcm'
seed_inner1 = (91,120)
seed_outer1 = (91,134)
seed_inner1_yx = (120, 91)
seed_outer1_yx = (134, 91)

class Label:
	def __init__(self, img, seed):
		self.img = img
		h, w = img.shape
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
				if x > 0 and self.img[y][x] == self.img[y][x-1]:
					self.union(self.yx2i(y, x - 1), self.yx2i(y, x))
				if y > 0 and self.img[y][x] == self.img[y-1][x]:
					self.union(self.yx2i(y - 1, x), self.yx2i(y, x))
		region = np.zeros(img.shape, dtype=np.uint8)
		# print 'size of region ', region.shape
		seedLabel = self.find(self.yx2i(self.seed[1], self.seed[0]))
		for y in range(self.nrows):
			for x in range(self.ncols):
				if self.find(self.yx2i(y, x)) == seedLabel:
					# print 'accessing ', x,y
					region[y][x] = 255
				else:
					region[y][x] = 0
		self.region = region
		return region
	def edgeDetect(self, region):
		h, w = region.shape
		edge = np.zeros(region.shape, dtype=np.uint8)
		for y in range(self.nrows):
			for x in range(self.ncols):
				# print x,y,region.shape
				if (x > 0 and region[y][x] != region[y][x-1]) \
				or (x <= w-2 and region[y][x] != region[y][x+1]) \
				or (y > 0 and region[y][x] != region[y-1][x]) \
				or (y <= h-2 and region[y][x] != region[y+1][x]):
					edge[y][x] = 255
		self.edge = edge
		return edge
	def getPixelValue(self, img, x, y, default):
		w, h = img.shape
		if x < 0 or x >= w or y < 0 or y >= h:
			return default
		else:
			return img[y][x]
	def erosion(self, oldimg, kernel):
		kernel = np.array(kernel, dtype=np.uint8)
		kh, kw = kernel.shape
		h, w = oldimg.shape
		erosed = np.zeros((h-kh+1, w-kw+1), dtype=np.uint8)
		for y in range(h-kh+1):
			for x in range(w-kw+1):
				yes = True
				for yy in range(kh):
					if not yes:
						break
					for xx in range(kw):
						# print y,yy,x,xx,oldimg.shape,h,kh,w,kw
						if oldimg[y+yy][x+xx] == 0 or kernel[yy][xx] == 0:
							yes = False
							break
				if yes:
					erosed[y][x] = 255
		return erosed


	def dilation(self, oldimg, kernel):
		kernel = np.array(kernel, dtype=np.uint8)
		kh, kw = kernel.shape
		h, w = oldimg.shape
		dilated = np.zeros((h-kh+1, w-kw+1), dtype=np.uint8)
		for y in range(h-kh+1):
			for x in range(w-kw+1):
				yes = False
				for yy in range(kh):
					if yes:
						break
					for xx in range(kw):
						# print y,yy,x,xx,oldimg.shape,h,kh,w,kw
						if oldimg[y+yy][x+xx] > 0 and kernel[yy][xx] > 0:
							yes = True
							break
				if yes:
					dilated[y][x] = 255
		return dilated
	def opening(self, oldimg, kernel):
		h, w = oldimg.shape
		erosed = self.erosion(oldimg, kernel)
		dilated = self.dilation(erosed, kernel)
		dilatedExpanded = np.zeros(oldimg.shape, dtype=np.uint8)
		kh, kw = np.array(kernel, dtype=np.uint8).shape
		print dilated.shape
		print h, w, kh, kw,  kh-1, h-kh+1, kw-1, w-kw+1
		dilatedExpanded[kh-1:h-kh+1, kw-1:w-kw+1] = dilated
		return dilatedExpanded

def medianFilter(oldimg, kernelsize):
	h, w = oldimg.shape
	k = kernelsize
	newimg = np.zeros((h-kernelsize+1, w-kernelsize+1), dtype=np.uint8)
	for y in range(h-kernelsize+1):
		for x in range(w-kernelsize+1):
			l = [0]*kernelsize*kernelsize
			for yy in range(kernelsize):
				for xx in range(kernelsize):
					l[yy*kernelsize+xx] = oldimg[y+yy][x+xx]
			l.sort()
			if kernelsize % 2 == 0:
				newimg[y][x] = (l[k*k/2-1] + l[k*k/2]) / 2
			else:
				# print l
				# print kernelsize, k, y, x, (k*k-1)/2, l[(k*k-1)/2]
				# print newimg[y][x]
				newimg[y][x] = l[(k*k-1)/2]
	return newimg


if __name__ == "__main__":
    ds = dicom.read_file(img1)
    # pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    # pylab.show()

    img = np.array(ds.pixel_array)
    print img.shape
    if TURN_ON_MEDIAN_FILTER:
    	img = medianFilter(img, 2)

    # test thresholding
    
    # outer = img[seed_outer1]
    # inner = img[seed_inner1]
    # old_threshold = (img[seed_outer1] + img[seed_inner1]) / 2
    # print outer, inner, old_threshold, outer < old_threshold, inner > old_threshold, img[seed_outer1] < old_threshold, img[seed_inner1] > old_threshold


    inner = 0
    for x in (-1,0,1):
        for y in (-1,0,1):
     	   inner += img[seed_inner1[1]+y][seed_inner1[0]+x]
    inner /= 9

    threshold = img[seed_outer1] / 4 + inner * 3 / 4
    print(img[seed_outer1_yx])
    print(inner)
    print(threshold)
    h, w = img.shape
    for x in range(w):
        for y in range(h):
            # print(x,y,img[y][x], threshold,  0 if img[y][x] < threshold else 65535)
            img[y][x] = 0 if img[y][x] < threshold else 65535

    imsave('16bitold.png', img)
    labelMatrix = Label(img, seed_inner1)
    region = labelMatrix.getImg()
    # edge = labelMatrix.edgeDetect(region)
  

     
    img8 = imread('16bitold.png')
    # region8 = imread('16bitnew.png')

    opened = labelMatrix.opening(region, [[1]*3]*3)
    openedLabelMatrix = Label(opened, seed_inner1)
    regionAfterOpening = openedLabelMatrix.getImg()


    backtorgb = cv2.cvtColor(img8,cv2.COLOR_GRAY2RGB)
    regiontorgb = cv2.cvtColor(region,cv2.COLOR_GRAY2RGB)
    # edge2rgb =  cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
    # erosed2rgb =  cv2.cvtColor(erosed,cv2.COLOR_GRAY2RGB)
    regionAfterOpening2rgb =  cv2.cvtColor(regionAfterOpening,cv2.COLOR_GRAY2RGB)

    cv2.circle(backtorgb,seed_inner1, 2, (255,255,0), -1)
    cv2.circle(backtorgb,seed_outer1, 1, (0,255,0), -1)


    # plt.imshow(regiontorgb)
    # plt.show()
    plt.imshow(regionAfterOpening2rgb)
    plt.show()
    imsave('old.png', backtorgb)
    imsave('8bitnew.png', regiontorgb)

