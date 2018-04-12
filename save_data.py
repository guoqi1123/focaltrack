import cv2
import flycapture2 as fc2
import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import json
import os, glob
import math

import matplotlib.pyplot as plt

from utils import *
import time

#notes
#  -did I mix up x and y again?  always worth checking

#debugging
import pdb

#multithreading
import threading
cameraThread = threading.Condition()
outsideThread = threading.Condition()
ftThread = threading.Condition()
displayThread = threading.Condition()
I_cache = 0
outside_I = 0
results = 0
ending_key = 'c'

#the range for different outputs, range set to NaN means auto-ranging
DEPTH_RANGE = [-1.1,-0.3]
FONT = cv2.FONT_HERSHEY_DUPLEX 
KEY_RANGE = {
	'raw' 			: [0,255],
	'gray' 			: [0,255],
	'test' 			: [0,255],
	'I_0' 			: [0,255],
	'I_1' 			: [0,255],
	'I_2' 			: [0,255],
	'Z'				: DEPTH_RANGE,
	'Zf'			: DEPTH_RANGE,
	'Z_crop'		: DEPTH_RANGE,
	'Z_cropdZdu'	: DEPTH_RANGE,
	'Z_cropw'		: DEPTH_RANGE,
	'estUnc'		: [-99999999,3],
}
class Camera:
	"""A class for the camera"""
	def __init__(self):
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

		self.t = []
		self.draw = None
		self.cache = {}
		self.vars = {}
		self.frames = 0
		self.total_frames = 1000
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

		self.I_cache = np.zeros(self.resolution+(2,), dtype = np.uint8)

		self.I = np.zeros(
			(self.total_frames,)+self.resolution+(2,), dtype = np.uint8
		)

	def run(self):
		global ending_key
		# The code to capture images
		t0 = time.time()
		while(self.frames<self.total_frames):
			self.t0 = time.time()
			self.grab_frame_and_process_ptg()
			self.regular_output()

			# obtain the input
			c = cv2.waitKey(1)
			if c != 255:
				ending_key = chr(c).lower()

			if ending_key == 'q':
				print("quitting")
				self.final_statistics()
				break

			self.t.append(time.time()-self.t0)

			# display frame rate in real time
			if np.mod(self.frames,1000)==0:
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("camera capture frame rate: (gross speed)", perf, " fps")
		
	def initialize_camera_ptg(self):
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
		self.cfg['camera_fps'] = p['abs_value']

	def decide_idx(self, new_image):
		# try to find out which image should be replaced
		err = np.zeros((self.I_cache.shape[2]))
		for i in range(self.I_cache.shape[2]):
			err[i] = np.mean(np.abs(self.I_cache[:,:,i] - new_image))
		self.idx = np.argmin(err)
		return 

	def naive_idx(self):
		self.idx = 1 - self.idx
		return

	"""imports a frame into """
	def process(self, new_frame):
		# Define the global variables
		global I_cache

		self.cache['raw'] = new_frame
		if len(new_frame.shape) > 2:
		# or new_frame.shape[2] != 1:
			self.cache['gray'] = cv2.cvtColor(self.cache['raw'], cv2.COLOR_BGR2GRAY)
		else:
			self.cache['gray'] = self.cache['raw'].copy()

		self.naive_idx()

		self.I_cache[:,:,self.idx] = self.cache['gray']
		self.I[self.frames,:,:,:] = self.I_cache

		self.frames += 1			
		return

	"""helper: grabs the frame and runs the processing stage""" 
	def grab_frame_and_process_ptg(self):
		# Using the point grey camera
		self.ptg.retrieve_buffer(self.im)
		img = np.array(self.im)
		img = scipy.misc.imresize(img, 1/self.cfg['downscale'])
		return self.process(img)
			
	"""computes the average FPS over the last __FPS_WINDOW frames"""
	def get_fps(self):
		__FPS_WINDOW = 4
		if len(self.t) < __FPS_WINDOW:
			return -1
		else:
			return __FPS_WINDOW/(1.0*sum(self.t[-1:-1 - __FPS_WINDOW:-1]))

	"""computes some performance statistics at the end"""
	def final_statistics(self):
		if len(self.t) < 2:
			print("too short, invalid statistics...")
			return

		t = np.array(self.t)
		fps = 1.0/t;
		print("""
			#####################################
			### CAMERA CAPTURE FPS STATISTICS ###
			#####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			#####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})

	def regular_output(self):
		self.cache['draw'] = tile_image(\
									I = [self.I_cache[:,:,0].astype(np.float32),\
										self.I_cache[:,:,1].astype(np.float32)], \
									rng = [KEY_RANGE['raw'], KEY_RANGE['raw']], \
									log = False, \
									title = "Camera Thread", \
								)
		cv2.imshow("Camera Thread", self.cache['draw'])
			
	"""destructor: free up resources when done"""
	def __del__(self):
		#TODO video writer stuff here
		cv2.destroyAllWindows()

if __name__ == "__main__":
	cam = Camera()
	cam.run()

	# save the data
	os.chdir('./raw_data/')
	lpickle = len(glob.glob('*.pickle'))
	fileName = os.path.join(\
		str(lpickle)+".pickle"
	)
	with open(fileName, 'wb') as f:
		data = {
				'I':		cam.I,
			}
		# dump the data into the file
		pickle.dump(data, f)