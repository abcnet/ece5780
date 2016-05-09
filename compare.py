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

TURN_ON_MEDIAN_FILTER = False

patients = [28, 30, 51, 100, 151, 177, 195, 198, 239, 244, 248, 254, 266, 272, 287, 332, 358, 395, 397, 398, 432, 434, 446, 457, 461]
done = 9
es = [[16, 17, 18], [10, 11, 12], [11, 12, 13], [13, 14, 15], [11, 12, 13], [13, 14, 15], [13, 14, 15], [10, 11, 12], [10, 11, 12]]

findPath = '../!Data/train/'
saxList = [[] for _ in range(done)]


def ballKernel(size):
	def frac(i, size):
		return (i + .5) / size
	def isInsideCircle(size, x, y):
		return 1 if (frac(x, size) - .5) ** 2 + (frac(y, size) - .5) ** 2 <= .25 else 0
	return [[isInsideCircle(size, x, y) for x in range(size)] for y in range(size)]

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
		region = np.zeros(self.img.shape, dtype=np.uint8)
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
						if kernel[yy][xx] > 0 and oldimg[y+yy][x+xx] == 0:
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
				segmentationPath = 'tiftest/' + str(patient) + '/sax_' + str(sax) + '.1-single-' + str(do-1) + '.png'
				# print segmentationPath
				if os.path.isfile(segmentationPath):
					# print timeFrame, imgPath, segmentationPath
					segmentation = imread(segmentationPath)
					# print segmentation
					if isinstance(segmentation[0][0], np.bool_):
						continue
					hasSegment = False
					sh, sw = segmentation.shape
					segmentSize = 0
					xs = 0
					ys = 0
					# print type(segmentation[0][0])
					for y in range(sh):
						
						for x in range(sw):
							if segmentation[y][x]:
								segmentSize += 1
								xs += x
								ys += y
								
					if segmentSize > 0:
						x_avg = xs / segmentSize
						y_avg = ys / segmentSize
						# print x, y, segmentation[y][x]
						# print segmentation
						
						# segmentationtorgb = cv2.cvtColor(np.array(segmentation, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
						# cv2.circle(segmentationtorgb, (x_avg, y_avg), 2, (255,255,0), -1)
						# plt.imshow(segmentationtorgb)
						# plt.show()

						print timeFrame, imgPath, segmentationPath
						ds = dicom.read_file(imgPath)
						dimg = np.array(ds.pixel_array)
						# dimgtorgb = cv2.cvtColor(np.array(dimg, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
						# plt.imshow(dimgtorgb)
						# plt.show()
						if TURN_ON_MEDIAN_FILTER:
							dimg = medianFilter(dimg, 2)

						inner = 0
						for x in (-1,0,1):
							for y in (-1,0,1):
								inner += dimg[y_avg][x_avg]
						inner /= 9
						threshold = inner * 2 / 3
						h, w = dimg.shape
						for x in range(w):
							for y in range(h):
								# print(x,y,dimg[y][x], threshold,  0 if dimg[y][x] < threshold else 65535)
								dimg[y][x] = 0 if dimg[y][x] < threshold else 65535
						labelMatrix = Label(dimg, (x_avg, y_avg))
						region = labelMatrix.getImg()
						opened = labelMatrix.opening(region, ballKernel(6))
						openedLabelMatrix = Label(opened, (x_avg, y_avg))
						regionAfterOpening = openedLabelMatrix.getImg()
						regionAfterOpening2rgb =  cv2.cvtColor(regionAfterOpening,cv2.COLOR_GRAY2RGB)
						plt.imshow(regionAfterOpening2rgb)
						plt.show()
# print saxList