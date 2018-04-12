import tensorflow as tf
import numpy as np
import cv2
import pdb
import pickle
import matplotlib.pyplot as plt
import pickle
import json
import os, glob
import copy
import scipy.misc 
from utils import *

def bp_filter_batch(I, low=0, high=10):
	I_back = I
	for i in range(I.shape[0]):
		for j in range(I.shape[3]):
			I_back[i,:,:,j] = bp_filter(I[i,:,:,j],low, high)
	return I_back

def bp_filter(I, low=0, high=10):
	# only keep the part with [low, high]
	I_f = np.fft.fft2(I)
	I_f = np.fft.fftshift(I_f)
	rows = I_f.shape[0]
	cols = I_f.shape[1]
	c_row = int(rows/2)
	c_col = int(cols/2)

	I_f[c_row-low+1:c_row+low,c_col-low+1:c_col+low] = 0
	I_f1 = copy.deepcopy(I_f)
	I_f1[c_row-high:c_row+high+1,c_col-high:c_col+high+1] = 0
	I_f = I_f - I_f1

	I_f = np.fft.ifftshift(I_f)
	I_back = np.fft.ifft2(I_f)
	I_back = np.abs(I_back)

	# I_disp = I_back - I_back.min()
	# I_disp = I_disp / I_disp.max()
	# cv2.imshow("1", I_disp)
	# cv2.waitKey(1)
	return I_back

def resize(I, rows, cols):
	I_back = np.zeros((I.shape[0],rows,cols,I.shape[3]))
	for i in range(I.shape[0]):
		for j in range(I.shape[3]):
			I_back[i,:,:,j] = scipy.misc.imresize(I[i,:,:,j],(rows, cols))

			I_disp = I_back[i,:,:,j] - I_back.min()
			I_disp = I_disp / I_disp.max()
			cv2.imshow("1", I_disp)
			cv2.waitKey(1)

	return I_back

# def bp_filter(I, low=0, high=10):
# 	# only keep the part with [low, high]
# 	I_f = np.fft.fft2(I)
# 	rows = I_f.shape[0]
# 	cols = I_f.shape[1]
	
# 	I_f[:,0:low] = 0
# 	I_f[0:low,:] = 0
# 	I_f[:,rows-low+1:rows]=0
# 	I_f[cols-low+1:cols,:]=0

# 	I_f[high+1:rows-high-1,high+1:cols-high-1] = 0
# 	I_back = np.fft.ifft2(I_f)
# 	I_back = np.abs(I_back)

# 	return I_back
