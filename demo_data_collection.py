#!/usr/bin/python2.7

import argparse

import cv2
import flycapture2 as fc2
import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import os

import matplotlib.pyplot as plt
from training_pyConfLensFlowNetFast import training_pyConfLensFlowNetFast

from utils import *
import time

#notes
#  -did I mix up x and y again?  always worth checking

#debugging
import pdb
import threading
myThread = threading.Condition()
I_cache = 0
resolution = 0
ending_key = 'c'

#the range for different outputs, range set to NaN means auto-ranging
DEPTH_RANGE = [-1.0, -0.25]
KEY_RANGE = {
	'raw' 			: [0,255],
	'gray' 			: [0,255],
	'test' 			: [0,255],
	'I_0' 			: [0,255],
	'I_1' 			: [0,255],
	'I_2' 			: [0,255],
	'Z'				: DEPTH_RANGE,
	'Z_crop'		: DEPTH_RANGE,
	'Z_cropdZdu'	: DEPTH_RANGE,
	'Z_cropw'		: DEPTH_RANGE,
	'estUnc'		: [-99999999,3],
}

KEYS_TO_TILE = ['ATA', 'ATA^-1', 'ATb', 'U']
class Camera(threading.Thread):
	"""A class for the camera"""
	def __init__(self):
		threading.Thread.__init__(self)
		#default configuraiton
		self.cfg = {
			#processing configuration knobs
			'wrapper': 'ptg', # Select the wrapper
			'downscale': 2, #downscaling of the image
			'camera' : 0, #the integer mapping to the camera you 
			#want to use.  typically 0, but depends on what's available
			'input_video' : None, #file name for a video file to 
			#use for debugging.  Will just loop through.  Can also 
			#use something like "path-to-images/%2d.tif" as the 
			#input here, that will show through all the images 
			#that match that string.
		}

		self.draw = None
		self.cache = {}
		self.vars = {}
		self.resolution = None

		#update index
		self.idx = 0
		
		#initialize camera
		self.cam = None
		self.initialize_camera_ptg()

		#initialize cache
		for k in ['gray']:
			self.cache[k] = np.zeros(self.resolution)
		
		#initialize computation
		self.Z_t = []

		self.sample_images = None

		self.data_type = tf.float32

		self.I_cache = np.zeros(self.resolution+(2,), dtype=np.float32)

	def run(self):
		global ending_key
		# The code to capture images
		while True:
			self.grab_frame_and_process_ptg()
			self.regular_output()

			c = cv2.waitKey(1) & 0xFF
			if chr(c).lower() == 'q'  or ending_key == 'q':
				ending_key = 'q'
				break
			if chr(c).lower() == 's':
				# switch the two images to always make the one
				# with smaller P (larger f) in front
				self.idx = 1 - self.idx
				tmp = self.I_cache[:,:,0]
				self.I_cache[:,:,0] = self.I_cache[:,:,1]
				self.I_cache[:,:,1] = tmp
				
		
	def initialize_camera_ptg(self):
		global resolution
		#access the point grey camera using flycap
		print(fc2.get_library_version())
		self.ptg = fc2.Context()
		print(self.ptg.get_num_of_cameras())
		self.ptg.connect(*self.ptg.get_camera_from_index(0))
		print(self.ptg.get_camera_info())

		m, f = self.ptg.get_video_mode_and_frame_rate()
		print(m, f)

		# print(c.get_video_mode_and_frame_rate_info(m, f))
		print(self.ptg.get_property_info(fc2.FRAME_RATE))
		p = self.ptg.get_property(fc2.FRAME_RATE)
		print(p)
		self.ptg.set_property(**p)
		self.ptg.start_capture()

		self.im = fc2.Image()
		# try to get the first image to 
		print([np.array(self.ptg.retrieve_buffer(self.im)).sum() for i in range(80)])
		img = np.array(self.im)
		print(img.shape, img.base)
		
		# downscaling the images
		img = scipy.misc.imresize(img, 1/self.cfg['downscale'])
		self.resolution = (
			img.shape[0],img.shape[1]
		)
		resolution = self.resolution

		self.cfg['camera_fps'] = p['abs_value']

	"""imports a frame into """
	def process(self, new_frame):
		global I_cache

		self.cache['raw'] = new_frame
		if len(new_frame.shape) > 2:
		# or new_frame.shape[2] != 1:
			self.cache['gray'] = cv2.cvtColor(self.cache['raw'], cv2.COLOR_BGR2GRAY)
		else:
			self.cache['gray'] = self.cache['raw'].copy()
		
		self.I_cache[:,:,self.idx] = self.cache['gray']
		self.idx = 1 - self.idx		

		# Lock the global variables, to updates those images
		myThread.acquire()
		I_cache = self.I_cache
		myThread.release()

		return
		
	"""helper: grabs the frame and runs the processing stage""" 
	def grab_frame_and_process_ptg(self):
		# Using the point grey camera
		self.ptg.retrieve_buffer(self.im)
		img = np.array(self.im)
		img = scipy.misc.imresize(img, 1/self.cfg['downscale'])
		return self.process(img)

	def regular_output(self):
		self.cache['draw'] = tile_image(\
									I = [self.I_cache[:,:,0],\
										self.I_cache[:,:,1]], \
									rng = [KEY_RANGE['raw'], KEY_RANGE['raw']], \
									log = False, \
									title = "Camera Thread", \
								)
		cv2.imshow("Camera Thread", self.cache['draw'])
			
	"""destructor: free up resources when done"""
	def __del__(self):
		#TODO video writer stuff here
		cv2.destroyAllWindows()

class DataCollector(threading.Thread):
	def __init__(self):
		global resolution
		threading.Thread.__init__(self)

		# initialize the variable that saves the image
		self.resolution = resolution
		self.I_cache = np.zeros(self.resolution+(2,), dtype=np.float32)
		self.I_total = np.empty(
			(0,)+self.resolution+(2,)
		)
		self.Loc_total = np.empty(
			(0,3,2)
		)

	def run(self):
		global I_cache
		global ending_key
		# take the image in
		while(True):
			in_key = 'r'
			while(in_key == 'r'):
				in_key = input(
					"ENTER THE LOCATION (mm) OF THE TRANSLATION STAGE!\n"
				)
				if ending_key == 'q':
					break
				loc_stage = float(in_key)
				
				in_key = input(
					"ENTER THE NUMBER (mm) ON THE TRANSLATION STAGE PANEL!'r' FOR REENTER\n"
				)
				if ending_key == 'q':
					break
				loc_plane = float(in_key)
				loc = (loc_plane - loc_stage)/1000

			if ending_key == 'q':
				break

			myThread.acquire()
			self.I_cache = I_cache
			myThread.release()

			I_tmp = np.empty((1,)+self.resolution+(2,))
			I_tmp[0,:,:,:] = self.I_cache
			self.I_total = np.concatenate(
				(self.I_total, I_tmp), axis = 0
			)

			# I hard codde here
			Loc = np.array([[[0,0],[0,0],[loc,loc]]])
			self.Loc_total = np.concatenate(
				(
					self.Loc_total,
					Loc
				),
				axis = 0
			)

			
		


# MAIN PROGRAM
a = Camera()
b = DataCollector()

a.start()
time.sleep(1)

b.start()

a.join()
b.join()

num_ave = 1
cfg = {
	'psf_func':		"pillbox",
	'Sigma':		.001,
	'pix_size':		5.86e-6*2,
	'f':			1e-1,
	'szx_sensor':	a.resolution[1],
	'szy_sensor':	a.resolution[0],
	'dP':			8000,
	'num_ave':		num_ave,
	'Z_0':			-1.38,
}

fileName = time.strftime("%Y%m%d%H%M%S", time.localtime())\
	+"_"+str(num_ave)

fullName = os.path.join(
	"./experiment_data/pyConfLensFlowNetFast",fileName+".pickle"
	)

with open(fullName, 'wb') as f:
	data = {
		'cfg':		cfg,
		'I':		b.I_total,
		'Loc':		b.Loc_total,
	}
	pickle.dump(data, f)
