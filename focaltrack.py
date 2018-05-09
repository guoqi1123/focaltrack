import serial
import argparse

import cv2
import flycapture2 as fc2
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.signal
import pickle
import json
import os, glob
import math
import copy

from utils import *
import time
from scipy import interpolate

#notes
#  -did I mix up x and y again?  always worth checking

#debugging
import pdb

#multithreading
import threading
displayThread = threading.Condition()
I_cache = 0
I_idx = 0 # update idx
outside_I = 0
results = 0
ending_key = 'c'
robust_mode = 'scanner_starter'

#the range for different outputs, range set to NaN means auto-ranging
DEPTH_RANGE = [-1.0,-0.3]#[-1.0,-0.3] # -1.0,-0.5
DEPTH_RANGE_f = [-2,-0]#[-1.0,-0.3] # -1.0,-0.5

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

KEYS_TO_TILE = ['ATA', 'ATA^-1', 'ATb', 'U']
class OutsideCamera(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.cfg = {
			#processing configuration knobs
			'wrapper': 'opencv', # Select the wrapper
			'downscale': 1, #downscaling of the image
			'camera' : 1, #the integer mapping to the camera you 
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
		self.resolution = None
		
		#initialize camera
		self.cam = None
		self.initialize_camera_opencv()
		
		#initialize cache
		for k in ['gray']:
			self.cache[k] = np.zeros(self.resolution)

		self.outside_I = np.zeros(self.resolution, dtype = np.uint8)

	def run(self):
		global ending_key
		# The code to capture images
		t0 = time.time()
		while True:
			self.t0 = time.time()
			self.grab_frame_and_process_opencv()
			
			# obtain the input
			displayThread.acquire()
			c = cv2.waitKey(1) & 0xFF
			displayThread.release()
			if c != 255:
				ending_key = chr(c).lower()

			if ending_key == 'q':
				print("quitting")
				self.final_statistics()
				break

			self.t.append(time.time()-self.t0)
			self.frames += 1

			# display frame rate in real time
			if np.mod(self.frames,1000)==0:
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("outside camera frame rate: (gross speed)", perf, " fps")

	def initialize_camera_opencv(self):
		#open camera using opencv wrapper
		self.cfg['cam_src'] = self.cfg['camera']
		if self.cfg['input_video'] is not None:
			self.cfg['cam_src'] = self.cfg['input_video']
		self.cam = cv2.VideoCapture(self.cfg['cam_src'])
		if not self.cam.isOpened():
			raise Exception("Could not open camera %s"%str(self.cfg['cam_src']))
		ry = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
		rx = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))	

		if self.cfg['downscale'] != 1:
			self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, ry/self.cfg['downscale'])
			self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, rx/self.cfg['downscale'])
			ry = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
			rx = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.resolution = [ry, rx]
		
	
		#try to run at least at 60 fps
		self.cam.set(cv2.CAP_PROP_FPS, 16)
		self.cfg['camera_fps'] = self.cam.get(cv2.CAP_PROP_FPS)

	def grab_frame_and_process_opencv(self):
		# Using the opencv camera
		r, raw = self.cam.read()
		if not r:
			if self.cfg['input_video'] is not None:
				self.cam = cv2.VideoCapture(self.cfg['cam_src'])
				r, raw = self.cam.read()
			else:
				raise Exception("Could not get image from camera '%s'"%str(self.cfg['cam_src']))	
		return self.process(raw)

	"""imports a frame into """
	def process(self, new_frame):
		# Define the global variables
		global outside_I

		self.cache['raw'] = new_frame
		if len(new_frame.shape) > 2:
		# or new_frame.shape[2] != 1:
			self.cache['gray'] = cv2.cvtColor(self.cache['raw'], cv2.COLOR_BGR2GRAY)
		else:
			self.cache['gray'] = self.cache['raw'].copy()
		
		self.outside_I = self.cache['gray'][150:300,170:-30:]

		# Lock the global variables, to updates those images
		outside_I = self.outside_I

		self.frames += 1			
		return

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
			### OUTSIDE CAMERA FPS STATISTICS ###
			#####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			#####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})

	def regular_output(self):
		self.cache['draw'] = tile_image(\
									I = [self.outside_I.astype(np.float32)], \
									rng = [KEY_RANGE['raw']], \
									log = False, \
									title = "Outside Camera", \
								)
		cv2.imshow("Outside Camera", self.cache['draw'])
			
	"""destructor: free up resources when done"""
	def __del__(self):
		if self.cam is not None:
			self.cam.release()
			self.cam = None
		#TODO video writer stuff here
		cv2.destroyAllWindows()

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

		self.t = []
		self.t0 = 0
		self.draw = None
		self.cache = {}
		self.vars = {}
		self.frames = 0
		self.resolution = None
		
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
		self.time_lapse = 0

	def run(self):
		global ending_key
		# The code to capture images
		t0 = time.time()
		while True:
			if self.t0 != 0:
				t1 = time.time()
				self.time_lapse = t1 - self.t0
				self.t0 = t1
			else:
				self.t0 = time.time()

			self.grab_frame_and_process_ptg()
			# self.regular_output()

			# obtain the input
			displayThread.acquire()
			c = cv2.waitKey(1) & 0xFF
			displayThread.release()
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
		global I_idx
		# try to find out which image should be replaced
		err = np.zeros((self.I_cache.shape[2]))
		for i in range(self.I_cache.shape[2]):
			err[i] = np.mean(np.abs(self.I_cache[:,:,i] - new_image))
		I_idx = np.argmin(err)
		return 

	def naive_idx(self):
		global I_idx
		I_idx = 1 - I_idx
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
		
		# # decide which idx of image should be updated
		# if self.frames > 2:
		# 	self.decide_idx(self.cache['gray'])
		# else:
		# 	self.naive_idx()

		# if self.time_lapse > 0.02:
		# 	# if the time is above threshold, we think it miss one frame
		# 	print("Time lapse (s): ", self.time_lapse)
		# else:
		# 	self.naive_idx()

		self.naive_idx()
		# self.decide_image()
		self.I_cache[:,:,I_idx] = self.cache['gray']

		# Lock the global variables, to updates those images
		I_cache = self.I_cache

		self.frames += 1			
		return

	def decide_image(self):
		err = [\
			np.mean(\
				np.abs(\
					self.cache['gray'].astype(np.float32)-\
					self.I_cache[:,:,0].astype(np.float32)
				)
			),
			np.mean(\
				np.abs(\
					self.cache['gray'].astype(np.float32)-\
					self.I_cache[:,:,1].astype(np.float32)
				)
			)
		]
		if err[0] < err[1]:
			idx = 0
		else:
			idx = 1
		if idx != I_idx:
			print("Switching Time lapse: ", self.time_lapse, err, I_idx, self.frames, )
			print(np.mean(self.cache['gray']),np.mean(self.I_cache[:,:,0]),\
				np.mean(self.I_cache[:,:,1]))			

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

class PulseCamProcessorTF(threading.Thread):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg, cfgf):
		threading.Thread.__init__(self)
		
		self.cfg = cfg
		self.cfgf = cfgf
		# resolution
		self.resolution = []
		for i in range(len(self.cfg)):
			self.resolution.append(
				(
					self.cfg[i]['szy_sensor'], 
					self.cfg[i]['szx_sensor']
				)
			)
			self.cfg[i]['P'] = 1/self.cfg[i]['f']

		# create an interpolation from a1 to offset
		x = self.cfgf[0]['a1_r']
		y = self.cfgf[0]['of_r']
		idx = np.argsort(x)
		x = x[idx]
		y = y[idx]
		self.cfgf[0]['of_rtck'] = interpolate.splrep(x,y)
		
		# # update the old parameters
		# keys = ['a0a1','b0','b1','w_bc','w_bc1','w_bc2']
		# for key in keys:
		# 	for i in range(len(self.cfg)):
		# 		self.cfg[i][key] = self.cfgf[i][key]
		
		self.required_outs = ['Z']
		self.vars_to_fuse = ['Z','conf','u_2']
		self.input_dict = {}

		# variables
		self.vars = []
		self.vars_align = {}
		self.vars_fuse = {}

		self.t = []
		self.draw = None
		self.cache = {}
		self.vars = []
		self.frames = 0
		self.frames_track = 0
		
		#initialize computation
		self.Z_t = []
		self.results = {}

		# the port
		self.Z_tgt = self.cfg[0]['a1']
		self.Z_tgt0 = self.cfg[0]['a1']
		for i in range(len(self.cfg)):
			self.cfg[i]['a0a1'] = self.cfg[i]['a0'] * self.cfg[0]['a1']

		self.offset_range = [\
			self.cfgf[0]['of_r'].min(),\
			self.cfgf[0]['of_r'].max()-1000\
		]
		self.inte = 0
		self.prev_err = 0
		self.offset = int((1/self.Z_tgt + 1.9)*1e4)
		if self.offset < self.offset_range[0]:
			self.offset = self.offset_range[0]
		elif self.offset > self.offset_range[1]:
			self.offset = self.offset_range[1]
		print("offset: ", self.offset)

		self.ser = serial.Serial()
		self.ser.port = "/dev/ttyUSB0" # may be called something different
		self.ser.baudrate = 9600 # may be different
		self.ser.open()
		# initialize the lens
		if self.ser.isOpen():
			string = "OF"+str(self.offset)+";"
			self.ser.write(string.encode())
			response = self.ser.read(self.ser.inWaiting())

		# we initialize with the tracking method to be passive
		self.track_methods = ['track_Z_pid']
		self.track_idx = 0
		self.robust_mode = 'scanner_starter'

		# tf graph
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.data_type = tf.float32

		# save the data from the camera
		self.I_cache = np.zeros(self.resolution[0]+(2,), dtype = np.uint8)
		
		# save the old data to keep the sequence
		self.old_data = {}
		self.old_num = 3
		self.old_idx = 0
		self.old_data['Z'] = [[] for i in range(self.old_num)]
		self.old_data['conf'] = [[] for i in range(self.old_num)]

		# make a video recorder
		self.build_graph()

	def run(self):
		global ending_key
		t0 = time.time()
		while True:
			self.t0 = time.time()
			self.process()
			self.robust_track_Z()

			# obtain the input
			displayThread.acquire()
			c = cv2.waitKey(1) & 0xFF
			displayThread.release()
			if c != 255:
				ending_key = chr(c).lower()

			# quit
			if ending_key == 'q':
				print("quitting")
				self.final_statistics()
				break

			# reset to the scanner
			if ending_key == 'r':
				self.robust_mode = 'scanner_starter'

			self.frames += 1
			self.frames_track += 1
			self.t.append(time.time()-self.t0)

			# display frame rate in real time
			if np.mod(self.frames,1000)==0:
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("FT avg performance: (gross speed)", perf, " fps")

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""
	def interp_a0(self, x_n, i):
		return interpolate.splev(x_n, self.cfgf[int(i)]['a0_rtck']).astype(np.float32)

	def interp_a1(self, x_n, i):
		return interpolate.splev(x_n, self.cfgf[int(i)]['a1_rtck']).astype(np.float32)


	def build_graph(self):
		with self.graph.as_default():
			# initialization of all variables
			I_init = np.zeros(
				self.resolution[0]+(self.cfg[0]['ft'].shape[2],),
				dtype = np.float32
			)
			self.I_in = tf.Variable(I_init)
			I = tf.Variable(I_init)

			self.a1_in = tf.Variable(1, dtype=tf.float32)
			a1 = tf.Variable(1, dtype=tf.float32)

			self.offset_in = tf.Variable(1, dtype=tf.float32)
			offset = tf.Variable(1, dtype=tf.float32)

			# quadratic correction
			px = []
			py = []
			ppx = self.cfgf[0]['ppx']
			ppy = self.cfgf[0]['ppy']
			for i in range(len(ppx)):
				o = tf.stack([(offset/10000)**k for k in range(len(ppx[i]))],0)
				px.append(tf.reduce_sum(np.flipud(ppx[i]) * o))

			for i in range(len(ppy)):
				o = tf.stack([(offset/10000)**k for k in range(len(ppy[i]))],0)
				py.append(tf.reduce_sum(np.flipud(ppy[i]) * o))

			# radial distortion
			xx,yy = np.meshgrid(\
				np.arange(self.resolution[0][1]),
				np.arange(self.resolution[0][0])
			)

			xx = (xx - (self.resolution[0][1]-1)/2)
			yy = (yy - (self.resolution[0][0]-1)/2)

			xr = [xx**(len(px)-k-1) * px[k] for k in range(len(px))]
			xr = tf.reduce_sum(tf.stack(xr,0),0)
			yr = [yy**(len(py)-k-1) * py[k] for k in range(len(py))]
			yr = tf.reduce_sum(tf.stack(yr,0),0)

			zr = xr * yr

			# radially confidence attenuation
			pxd = tf.stack([px[k]*(len(px)-1-k) for k in range(len(px)-1)],-1)
			rxs = tf.py_func(np.roots, [pxd], tf.float32)
			xxr = xx/tf.abs(rxs[0])
			yyr = yy/tf.abs(rxs[0])
			# confr = 0.9985 + 0.0015/(1+tf.exp((tf.sqrt(xxr**2+yyr**2)-1)))
			confrr = tf.sqrt(xxr**2+yyr**2)-1
			confrr = tf.nn.relu(confrr)
			confr = 0.998 + 0.002/(1+10*confrr)

			I_batch = []
			I_lap_batch = []

			Z_0 = []
			a0a1 = []
			a0 = []
			ra0_1 = []
			ra0_2 = []
			ra1_1 = []
			ra1_2 = []
			Z_0f = []
			a0f = []
			a1f = []

			gauss = []
			ext_f = []
			ft = []
			fave = []

			I_t = []
			I_lap = []

			u_1 = []
			u_2 = []
			u_3 = []
			u_4 = []
			u_2f = []
			u_3f = []
			u_4f = []
			self.a1f = []

			# depth computation
			# I used to be (960, 600, 2or 3) we change it to (2or3, 960, 600) 
			tmp_I = tf.stack(\
				[dIdt(I, self.cfg[0]['ft']), \
				dIdt(I, self.cfg[0]['fave'])]
			)
			# tmp_I = tf.transpose(I, perm=[2,0,1])
			for i in range(len(self.cfg)):
				# initialize variables				
				"""Input parameters"""
				Z_0.append(tf.constant(self.cfg[i]['Z_0'], dtype = tf.float32))
				a0a1.append(tf.constant(self.cfg[i]['a0a1'], dtype = tf.float32))
				a0.append(a0a1[i]/a1)

				Z_0f.append(tf.constant(self.cfgf[i]['Z_0'], dtype= tf.float32))

				# interpolate a0 and a1
				a0f.append(tf.py_func(\
					self.interp_a0, \
					[offset, tf.constant(i, dtype=tf.float32)], \
					tf.float32
				))
				a1f.append(tf.py_func(\
					self.interp_a1, \
					[offset, tf.constant(i, dtype=tf.float32)], \
					tf.float32
				))

				# # radial distortion
				# xx,yy = np.meshgrid(\
				# 	np.arange(self.resolution[i][1]),
				# 	np.arange(self.resolution[i][0])
				# )

				# xx = (xx - (self.resolution[i][1]-1)/2)/self.resolution[i][0]
				# yy = (yy - (self.resolution[i][0]-1)/2)/self.resolution[i][0]
				# rr = np.sqrt(xx**2 + yy**2)
				# r = tf.constant(rr, dtype=tf.float32)
				# ra0_1.append(tf.Variable(self.cfgf[i]['ra0_1'], dtype=tf.float32))
				# ra0_2.append(tf.Variable(self.cfgf[i]['ra0_2'], dtype=tf.float32))
				# ra1_1.append(tf.Variable(self.cfgf[i]['ra1_1'], dtype=tf.float32))
				# ra1_2.append(tf.Variable(self.cfgf[i]['ra1_2'], dtype=tf.float32))

				# # pdb.set_trace()
				# a0f_r = a0f[i] + ra0_1[i] * r + ra0_2[i] * (r**2)
				# a1f_r = a1f[i] + ra1_1[0] * r + ra1_2[0] * (r**2)

				a0f_r = a0f[i]
				a1f_r = a1f[0]

				gauss.append(tf.constant(self.cfg[i]['gauss'], dtype = tf.float32))
				ext_f.append(tf.constant(self.cfg[i]['ext_f'], dtype = tf.float32))
				ft.append(tf.constant(self.cfg[i]['ft'], dtype = tf.float32))
				fave.append(tf.constant(self.cfg[i]['fave'], dtype = tf.float32))

				# multi resolution
				I_t.append(tmp_I[0,:,:])
				tmp_I_blur = dIdx_batch(tmp_I, gauss[i])
				I_lap.append(tmp_I[1,:,:] - tmp_I_blur[1,:,:])

				if i < len(self.cfg)-1:
					tmp_I = tf.squeeze(\
						tf.image.resize_bilinear(\
							tf.expand_dims(
								tmp_I_blur,-1
							),
							self.resolution[i+1],
							align_corners = True
						),[-1]
					)

				# unwindowed version
				u_1.append(I_lap[i])
				u_2.append(I_t[i])
				u_3.append(a0[i] * a1 * u_1[i])
				u_4.append(-u_2[i] + a0[i] * u_1[i] + 1e-5)
				u_3f.append(a0f_r * a1f_r * u_1[i])
				u_4f.append(-u_2[i] + a0f_r * u_1[i] + 1e-5)
				ext_f[i] = tf.expand_dims(\
					tf.transpose(ext_f[i], perm=[1,2,0]),-2
				)
				u_2[i] = tf.expand_dims(tf.expand_dims(u_2[i],0),-1)
				u_3[i] = tf.expand_dims(tf.expand_dims(u_3[i],0),-1)
				u_4[i] = tf.expand_dims(tf.expand_dims(u_4[i],0),-1)
				u_3f[i] = tf.expand_dims(tf.expand_dims(u_3f[i],0),-1)
				u_4f[i] = tf.expand_dims(tf.expand_dims(u_4f[i],0),-1)
				u_2[i] = tf.nn.conv2d(u_2[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_2f.append(u_2[i])
				u_3[i] = tf.nn.conv2d(u_3[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_4[i] = tf.nn.conv2d(u_4[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_3f[i] = tf.nn.conv2d(u_3f[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_4f[i] = tf.nn.conv2d(u_4f[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				
				#save references to required I/O]
				self.vars.append({})
				# self.vars[i]['I'] = I_batch[i]
				
				# unwindowed version
				self.vars[i]['u_1'] = u_1[i]
				self.vars[i]['u_2'] = u_2[i]
				self.vars[i]['u_2f'] = u_2f[i]
				self.vars[i]['u_3'] = u_3[i]
				self.vars[i]['u_4'] = u_4[i]
				self.vars[i]['u_3f']= u_3f[i]
				self.vars[i]['u_4f']= u_4f[i]

			# align depth and confidence maps
			self.align_maps_ext(['u_2','u_2f','u_3','u_4','u_3f','u_4f'])

			# flatten the u_3, u_4, u_4f to enable faster computation
			for key in ['u_2','u_2f','u_3','u_4','u_3f','u_4f']:
				self.vars_align[key] = tf.reshape(\
					self.vars_align[key],
					[-1, len(self.cfg)*self.cfg[0]['ext_f'].shape[0]]
				)

			# compute the aligned version of Z
			self.vars_align['Z'] = \
				self.vars_align['u_3'] / (self.vars_align['u_4'] + 1e-5)

			self.vars_align['Zf'] = \
				self.vars_align['u_3f'] / (self.vars_align['u_4f']+ 1e-5)

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[0]['conf_func']+'()')

			# fusion
			self.softmax_fusion()

			# radial correction
			self.vars_fuse['Zf'] /= zr
			self.vars_align['conf_non'] = self.vars_align['conf']
			self.vars_fuse['conf_non'] = self.vars_fuse['conf']
			self.vars_fuse['conf'] *= confr
			self.vars_fuse['conff'] *= confr

			# averaging the confidence to remove noise
			conf_ave = tf.stack(\
				[self.vars_fuse['conf'],self.vars_fuse['conf_non'], self.vars_fuse['conff']]
			)
			conf_ave = dIdx_batch(conf_ave, gauss[0])
			# # threshold the confidence by I_t
			# flag = tf.cast(tf.less(10., self.vars_fuse['u_2']), tf.float32)
			self.vars_fuse['conf'] = conf_ave[0,:,:] #* flag
			self.vars_fuse['conf_non'] = conf_ave[1,:,:] #* flag
			self.vars_fuse['conff'] = conf_ave[2,:,:]

			# reshape back all the aligned version
			for key in ['u_2','u_3','u_4','u_3f','u_4f','Z','Zf','conf','conf_non','conff']:
				self.vars_align[key] = tf.reshape(\
					self.vars_align[key],
					self.resolution[0]+(-1,)
				)

			# 
			self.valid_windowed_region_fuse()

			# remove the bias
			for key in ['Z']:
				self.vars_fuse[key] = -self.vars_fuse[key]
				self.vars_align[key] = -self.vars_align[key]

			for key in ['Zf']:
				self.vars_fuse[key] = -self.vars_fuse[key]
				self.vars_align[key] = -self.vars_align[key]

			# # temporal averaging
			# conf_old = tf.Variable(self.vars_fuse['conf'], dtype=tf.float32)
			# l = 0.91
			# self.vars_fuse['conf'] = l*self.vars_fuse['conf']+(1-l)*conf_old
			# conf_old = self.vars_fuse['conf']

			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			self.input_data = tf.group(\
				I.assign(self.I_in), a1.assign(self.a1_in), offset.assign(self.offset_in)
			)

			#do not add anything to the compute graph 
			#after this line
			init_op = tf.global_variables_initializer()
			self.session.run(init_op)

	def align_maps_ext(self, vars_to_fuse = None):
		# this function aligns different
		# res into the same one
		if vars_to_fuse == None:
			vars_to_fuse = self.vars_to_fuse
		shp = self.resolution[0]
		for var in vars_to_fuse:
			self.vars_align[var] = []
			for i in range(len(self.cfg)):
				# align depth map and confidence
				self.vars_align[var].append(\
					tf.image.resize_bilinear(\
						tf.expand_dims(
							self.vars[i][var],0
						),\
						shp,
						align_corners = True
					)\
				)
			# concatenate the depth maps and confidence
			self.vars_align[var] = tf.squeeze(
				tf.concat(self.vars_align[var], 3), [0]
			)
		return 

	def softmax_fusion(self):
		ws = tf.nn.softmax(self.vars_align['conf']*1e10)

		# fuse the results using softmax
		for var in self.vars_to_fuse:
			self.vars_fuse[var] = \
				tf.reshape(
					tf.reduce_sum(
						self.vars_align[var]*ws,
						[1]
					), self.resolution[0]
				)


		conf_flat = self.vars_align['conff']

		ws = tf.nn.softmax(conf_flat*1e10)

		# fuse the results using softmax
		for var in self.vars_to_fuse:
			self.vars_fuse[var+'f'] = \
				tf.reshape(
					tf.reduce_sum(
						self.vars_align[var+'f']*ws,
						[1]
					), self.resolution[0]
				)

		return 

	def w3_baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability,
		# use the working range to cut the confidence
		# and use weight for each layer
		# the windowed flag indicates whether we compute windowed
		w_bc = [] # the weight of baseline confidence for each layer
		w_bc1 = []
		w_bc2 = []
		for i in range(len(self.cfg)):
			# weights
			w_bc.append(\
				tf.constant(
					self.cfg[i]['w_bc'],
					dtype = tf.float32
				)	
			)
			w_bc1.append(\
				tf.constant(
					self.cfg[i]['w_bc1'],
					dtype = tf.float32
				)	
			)
			w_bc2.append(\
				tf.constant(
					self.cfg[i]['w_bc2'],
					dtype = tf.float32
				)
			)


		w_bc = tf.concat(w_bc,0)
		w_bc1 = tf.concat(w_bc1,0)
		w_bc2 = tf.concat(w_bc2,0)

		u_3 = self.vars_align['u_3']
		u_32 = u_3**2
		u_4 = self.vars_align['u_4']
		u_42 = u_4**2

		self.vars_align['conf'] = (u_42 + 1e-20)/\
			tf.sqrt(\
				tf.multiply(u_32, w_bc) + \
				tf.multiply(u_42, w_bc1) + \
				tf.multiply(u_3*u_4, w_bc2) + \
				u_42**2 + 1e-10				 
			)

		self.vars_align['conff'] = self.vars_align['conf']

		return

	def valid_windowed_region_fuse(self):
		# cut out the bad part
		vars_to_cut = [\
			'Z','conf','conf_non','Zf','conff'\
		]
		rows_cut = int(\
			((self.cfg[-1]['gauss'].shape[0]-1)/2+\
			(self.cfg[-1]['ext_f'].shape[1]-1)/2)*\
			self.resolution[0][0]/self.resolution[-1][0]
		)
		cols_cut = int(\
			((self.cfg[-1]['gauss'].shape[1]-1)/2+\
			(self.cfg[-1]['ext_f'].shape[2]-1)/2)*\
			self.resolution[0][1]/self.resolution[-1][1]
		)
		rows = self.cfg[0]['szx_sensor']
		cols = self.cfg[0]['szy_sensor']

		for var in vars_to_cut:
			self.vars_fuse[var] = \
				self.vars_fuse[var][
					cols_cut:cols-cols_cut,
					rows_cut:rows-rows_cut
				]

			self.vars_align[var] = \
				self.vars_align[var][
					cols_cut:cols-cols_cut,
					rows_cut:rows-rows_cut,
					:
				]

		return 

	"""imports a frame into """
	def process(self):
		global results
		# input the data
		self.input_dict[self.I_in] = I_cache		
		self.input_dict[self.a1_in] = self.cfg[0]['a1']
		self.input_dict[self.offset_in] = self.offset

		self.session.run(self.input_data, self.input_dict)

		# run it
		self.image_to_show = ['Z','Zf','conf','u_2','conf_non']
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars_fuse[k]
		self.results = self.session.run(res_dict)

		# if self.robust_mode == 'tracker':
		# 	# keep the correct sequence
		# 	self.keep_sequence()

		self.swap_frames()

		results = self.results

		return

	def keep_sequence(self):
		# keep the correct sequence
		if self.frames_track > self.old_num:
			self.results['Z'], self.results['conf'], \
			self.results['Zf'], self.results['conff'] \
			= self.robust_depth(\
				self.old_data['Z'],\
				self.results['Z'],\
				self.results['conf'],\
				self.results['Zf'],\
				self.results['conff'],
			)

		self.old_data['Z'][self.old_idx] = self.results['Z']
		self.old_data['conf'][self.old_idx] = self.results['conf']
		self.old_idx = np.mod(self.old_idx+1, self.old_num)

	def swap_frames(self):
		global I_cache
		global ending_key
		global I_idx
		if ending_key == 's':
			# for i in range(len(self.old_data['Z'])):
			# 	self.old_data['Z'][i] = self.results['Zf']
			ending_key = 'c'

			# swap the image pair
			tmp = I_cache[:,:,0]
			I_cache[:,:,0] = I_cache[:,:,1]
			I_cache[:,:,1] = I_cache[:,:,0]
			I_idx = 1 - I_idx

	def robust_depth(self, old, new, conf, newf, conff):
		# robustly find the correct depth map that is consistent with 
		# the previous result
		err = np.array(\
			[\
				min([\
					np.mean(\
						np.abs(old[i] - new)\
					)\
					for i in range(len(old))\
				])
				, \
				min([\
					np.mean(\
						np.abs(old[i] - newf)\
					)\
					for i in range(len(old))
				])
			]\
		)
		depth = [new, newf]
		confs = [conf, conff]
		if np.argmin(err) == 2:
			# swap the image pair
			tmp = I_cache[:,:,0]
			I_cache[:,:,0] = I_cache[:,:,1]
			I_cache[:,:,1] = I_cache[:,:,0]
			I_idx = 1 - I_idx
		return depth[np.argmin(err)],confs[np.argmin(err)],\
			depth[np.argmax(err)],confs[np.argmax(err)]

	def robust_track_Z(self):
		global robust_mode
		# crop out high confidence regions
		self.track_thre = 0.9

		# averaging confidence
		conf = copy.deepcopy(self.results['conf_non'])
		Z = copy.deepcopy(self.results['Z'])
		
		# check number of high conf pixels
		self.Z_high = Z[np.where(conf > self.track_thre)]
		
		# find the median
		if self.Z_high.flatten().shape[0] > 0:
			# set an roi
			# up = 75
			# lf = 120
			# up = 1
			# lf = 80
			# Z = Z[up:-up,lf:-lf]
			# conf = conf[up:-up,lf:-lf]
			# Z_high_roi = Z[np.where(conf > self.track_thre)]
			# self.Z_tgt = np.abs(np.percentile(Z_high_roi,30))
			# pdb.set_trace()
			self.Z_tgt = np.abs(np.median(self.Z_high))
			# self.Z_tgt = np.abs(np.percentile(self.Z_high, 50))
			# close_thre = 0.45
			# if self.Z_tgt < close_thre:
			# 	self.Z_tgt = np.abs(np.percentile(self.Z_high, 50))
			# 	if self.Z_tgt >= close_thre:
			# 		self.Z_tgt = close_thre
				# pdb.set_trace()
		else:
			self.Z_tgt = np.nan

		# # do the flip only when needed
		# if self.robust_mode == 'nothing':
		# 	# averaging confidence
		# 	conff = self.results['conff']
		# 	Zf = self.results['Zf']

		# 	# check number of high conf pixels
		# 	self.Zf_high = Zf[np.where(conff > self.track_thre)]
			
		# 	# find the median
		# 	if self.Zf_high.flatten().shape[0] > 0:
		# 		self.Zf_tgt = np.abs(np.median(self.Zf_high))
		# 	else:
		# 		self.Zf_tgt = np.nan

		# 	print(self.Z_high.flatten().shape[0], \
		# 		self.Z_tgt, \
		# 		self.Zf_tgt, \
		# 		self.cfg[0]['a1'], \
		# 		self.offset, \
		# 		self.is_object(), \
		# 		self.batch_idx\
		# 	)

		# follow the function
		eval('self.'+self.robust_mode+'()')
		robust_mode = self.robust_mode

		return

	def scanner_starter(self):
		# initialize the scanner
		self.offset = self.a1_to_offset(self.cfg[0]['a1'])
		self.cfg[0]['a1'] = self.offset_to_a1(self.offset)
		self.cfgf[0]['a1'] = self.offset_to_a1_f(self.offset)
		
		# find the offset with most high confidence pixels
		self.batch_size = 5
		self.batch_idx = 0
		self.max_pix = 0
		self.max_offset = self.offset_range[0]
		self.max_frame = self.frames
		self.scanner_step = 0.01

		self.control_lens()
		self.robust_mode = 'scanner_iter'
		return

	def scanner_iter(self):
		global ending_key
		pix_num = self.Z_high.flatten().shape[0]
		if self.is_object():
			if pix_num > self.max_pix:
				# if we find a frame with more pixels
				self.max_pix = pix_num
				self.max_offset = self.offset
				self.max_frame = self.frames
				self.batch_idx = 0
			else:
				self.batch_idx += 1

		# check if we already find the maximum
		if self.batch_idx >= self.batch_size:
			# if we already find the frame with max high conf pixels
			self.offset = self.max_offset 
			self.cfg[0]['a1'] = self.offset_to_a1(self.offset)
			self.control_lens()

			self.robust_mode = 'tracker_starter'
		else:
			# linear change of a1
			self.cfgf[0]['a1'] = self.cfgf[0]['a1'] + self.scanner_step 
			# if self.cfgf[0]['a1'] > -DEPTH_RANGE_f[0]:
			# 	self.cfgf[0]['a1'] = -DEPTH_RANGE_f[0]
			# 	self.scanner_step = -self.scanner_step
			# elif self.cfgf[0]['a1'] < -DEPTH_RANGE_f[1]:
			# 	self.cfgf[0]['a1'] = -DEPTH_RANGE_f[1]
			# 	self.scanner_step = -self.scanner_step

			# thresholding offset
			self.offset = self.a1_to_offset_f(self.cfgf[0]['a1'])
			if self.offset < self.offset_range[0]:
				self.offset = self.offset_range[0]
				self.scanner_step = -self.scanner_step
			elif self.offset > self.offset_range[1]:
				self.offset = self.offset_range[1]
				self.scanner_step = -self.scanner_step
			
			# renew the a1
			self.cfgf[0]['a1'] = self.offset_to_a1_f(self.offset)
			self.cfg[0]['a1'] = self.offset_to_a1(self.offset)
			# print(self.offset)
			# print(self.cfgf[0]['a1'])
			# print(self.cfg[0]['a1'])
			# print(self.offset)
			# print('scanning')

			self.control_lens()

		return

	def tracker_starter(self):
		# clean out the previous data
		self.prev_err = 0
		self.inte = 0

		# prepare data
		# go to tracking
		self.frames_track = 0
		self.old_data = {}	
		self.old_idx = 0
		self.old_data['Z'] = [[] for i in range(self.old_num)]
		self.old_data['conf'] = [[] for i in range(self.old_num)]

		# stablize the tracking
		self.stab_num = 0
		self.stab_idx = 0

		self.robust_mode = 'stablizer'
		return

	def stablizer(self):
		if self.stab_idx < self.stab_num:
			self.stab_idx += 1
		else:
			self.robust_mode = 'tracker'

	def tracker(self):
		global ending_key
		# eval('self.'+self.track_methods[self.track_idx]+'()')
		if not self.is_object():
			# if lose the object, change back to scanner 
			self.robust_mode = 'scanner_starter'
			# ending_key = 's'
		else:
			# do tracking
			eval('self.'+self.track_methods[self.track_idx]+'()')
		return 

	def passive_Z(self):
		# if not enough pixels and in pid mode, change to scanner mode
		# this mode fix the a1 and offset
		return

	def track_Z_pid(self):
		# if not enough pixels and in pid mode, change to scanner mode
		# compute the pid
		self.err = self.cfg[0]['a1'] - self.Z_tgt
		self.inte += self.err
		self.der = self.err - self.prev_err 

		# Kp = -0.05
		# Ki = -0.001
		# Kd = -0.01

		Kp = -0.4
		Ki = -0.0001
		Kd = -0.02

		dZ = Kp*self.err + Ki*self.inte + Kd*self.der
		Z_set = self.cfg[0]['a1'] +dZ
		self.prev_err = self.err

		# thresholding
		if Z_set > -DEPTH_RANGE[0]:
			Z_set = -DEPTH_RANGE[0]
		elif Z_set < -DEPTH_RANGE[1]:
			Z_set = -DEPTH_RANGE[1]

		self.offset = self.a1_to_offset(Z_set)

		if self.offset < self.offset_range[0]:
			self.offset = self.offset_range[0]
		elif self.offset > self.offset_range[1]:
			self.offset = self.offset_range[1]
		
		self.cfg[0]['a1'] = self.offset_to_a1(self.offset)
		self.cfgf[0]['a1'] = self.offset_to_a1_f(self.offset)
		# print(Z_set, self.offset, self.Z_tgt)
		self.control_lens()
		return 

	def offset_to_a1(self, offset):
		return 1/(self.cfg[0]['b0']*offset+self.cfg[0]['b1'])

	def a1_to_offset(self, a1):
		if a1 == 0:
			pdb.set_trace()
		return int((1/a1 - self.cfg[0]['b1'])/self.cfg[0]['b0'])

	def offset_to_a1_f(self, offset):
		return interpolate.splev(offset, self.cfgf[0]['a1_rtck'])

	def a1_to_offset_f(self, a1):
		return int(interpolate.splev(a1, self.cfgf[0]['of_rtck']))

	def control_lens(self):
		if self.ser.isOpen():
			string = "OF"+str(self.offset)+";"
			self.ser.write(string.encode())
			response = self.ser.read(self.ser.inWaiting())
			# print(response)
		return

		"""computes the average FPS over the last __FPS_WINDOW frames"""

	def is_object(self):
		# check if there is object in the image
		self.N_pix = 100

		if self.Z_high.flatten().shape[0] < self.N_pix or np.isnan(self.Z_tgt):
			return False
		elif self.Z_tgt > -DEPTH_RANGE[0]:
			return False
		elif self.Z_tgt < -DEPTH_RANGE[1]:
			return False
		else:
			return True

	def quad_comp(self):
		row = self.resolution[0][0]
		col = self.resolution[0][1]
		x = np.arange(col) - (col-1)/2
		y = np.arange(row) - (row-1)/2
		xx, yy = np.meshgrid(x, y)
		d = 2.5e-5
		# d = 0
		self.quad = d*np.sqrt(xx**2 + yy**2) + 1
		return

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
			####################################
			### FT PROCESSING FPS STATISTICS ###
			####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})
			
	"""destructor: free up resources when done"""
	def __del__(self):
		# self.video_writer.release()
		cv2.destroyAllWindows()
		

MAX_VAL = 255.0
from scipy.sparse import csr_matrix

# From: https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)

def rgb2yuv(im):
    return (np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET)

def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))

def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx] 
    return valid_idx, locs

class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)
        
    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True) 
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)
        
    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)
    
    def slice(self, y):
        return self.S.T.dot(y)
    
    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out += blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))

import sklearn.cluster
class Display(threading.Thread):
	def __init__(self, cfg, cfgf):
		threading.Thread.__init__(self)
		self.cfg = cfg
		self.cfgf = cfgf
		# resolution
		self.resolution = []
		for i in range(len(self.cfg)):
			self.resolution.append(
				(
					self.cfg[i]['szy_sensor'], 
					self.cfg[i]['szx_sensor']
				)
			)
			self.cfg[i]['P'] = 1/self.cfg[i]['f']

		self.depth_data = {}
		self.cache = {}
		self.t = []
		self.frames = 0
		self.track_count = 0
		self.track_num = 50
		self.show_modes = ['origin','partial BFS','full BFS']
		self.show_mode = 0
		# self.clustering = sklearn.cluster.KMeans(2)

	def run(self):
		global ending_key
		t0 = time.time()
		cv2.namedWindow("Focal Track Demo", cv2.WINDOW_NORMAL)
		while True:
			self.t0 = time.time()
			self.process()
			self.iccv_output()

			# obtain the input
			displayThread.acquire()
			c = cv2.waitKey(1) & 0xFF
			displayThread.release()
			if c != 255:
				ending_key = chr(c).lower()
			
			# quit
			if ending_key == 'q':
				print("quitting")
				self.final_statistics()
				break
			# capture the data
			if ending_key == '\n':
				# collect the selected data
				directory = "./models/"
				lavi = len(glob.glob(\
					directory + "*.pickle"
				))
				fileName = "./models/"\
					+ str(lavi) + ".pickle"
				with open(fileName,'wb') as f:
					pickle.dump(self.depth_data, f)
				ending_key = 'c'

			# screen shot
			if ending_key == 't':
				scipy.misc.imsave('./models/demoshot/plain.png', \
					(self.cache['draw']*255).astype(np.uint8)
				)
				ending_key = 'c'

			if ending_key == ' ':
				self.show_mode += 1 
				self.show_mode = np.mod(self.show_mode, len(self.show_modes))
				ending_key = 'c'

			self.frames += 1
			self.t.append(time.time()-self.t0)

			# display frame rate in real time
			if np.mod(self.frames,1000)==0:
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("display frame rate: (gross speed)", perf, " fps")

	def process(self):
		self.I_cache = I_cache
		# self.outside_I = outside_I
		self.results = results
		return

	def to_show(self):
		if robust_mode == 'tracker':
			self.track_count += 1
		else:
			self.track_count = 0
		
		# only display when tracking for some time
		if self.track_count > self.track_num:
			return True
		else:
			return False

	def demo_output(self):	
		# backup the data for saving
		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		
		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.I_cache[:,:,1].astype(np.float32)
		self.results['I_1'] = self.I_cache[:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		# create the image to draw
		Z = self.prep_for_draw_demo(I = Z, message='Depth (C>0.995)', rng=KEY_RANGE['raw'])
		I = self.prep_for_draw_demo(I = self.results['I_0'], message='Input image (1 of 2)', rng=KEY_RANGE['raw'])
		self.cache['draw'] = np.concatenate((I,Z), axis=1)

		# new shape
		ncol = I.shape[1] + Z.shape[1]

		# in-focus plane
		flegend = np.zeros((20, ncol), dtype=np.float32)
		fplane = np.zeros((10, ncol), dtype=np.float32)
		Z_f = self.cfg[0]['a1']
		loc = int((Z_f + DEPTH_RANGE[1])/(DEPTH_RANGE[1]-DEPTH_RANGE[0])*ncol)
		fplane[:, loc-2:loc+2] = 1
		fplane = self.prep_for_draw_demo(I = fplane, rng=[0,1])
		flegend = self.prep_for_draw_demo(I = flegend, rng=[0,1])
		
		t_s = 0.5
		t_h = int(20*t_s)
		flegend = cv2.putText(\
			flegend, \
			"Focus", \
			(loc-20,t_h+3), \
			FONT, \
			t_s, \
			(1,1,1)\
		)

		# make a color bar
		cbar_row = 10
		seg = 100
		# color bar
		cbar = np.matlib.repmat(np.linspace(100,0,num=seg), cbar_row, 1)
		# pseudo confidence
		cpseudo = np.ones((cbar_row, seg))
		# give the pseudo color
		cbar = (self.pseudo_color_Z(cbar, cpseudo, 0, 100, 0) * 255).astype(np.uint8)
		cbar = scipy.misc.imresize(cbar, (cbar_row, ncol)).astype(np.float32)/255

		nums = np.linspace(0.3,1.1,9)
		for i in range(nums.shape[0]-1):
			loc = int(ncol / (nums.shape[0]-1) * i)
			cbar[-6::,loc,0] = 0
			cbar[-6::,loc,1] = 0
			cbar[-6::,loc,2] = 0

		# make the number bar
		nbar = np.zeros((30, cbar.shape[1]),dtype=np.float32)
		nbar = self.prep_for_draw_demo(I = nbar, rng=[0,100])
		t_s = 0.5
		t_h = int(20*t_s)
		nbar = cv2.putText(\
			nbar, \
			"Depth(m)",\
			(0, t_h+5), \
			FONT, \
			t_s, \
			(1,1,1)\
		)
		for i in range(nums.shape[0]-2):
			loc = int(ncol / (nums.shape[0]-1) * (i+1))
			nbar = cv2.putText(\
				nbar, \
				str(nums[i+1]), \
				(loc-13, t_h+5), \
				FONT, \
				t_s, \
				(1,1,1)\
			)

		# outside color
		I_o = scipy.misc.imresize(self.outside_I, \
			(int(ncol/self.outside_I.shape[1]*self.outside_I.shape[0]), ncol))
		I_o = self.prep_for_draw_demo(I = I_o, rng=KEY_RANGE['raw'])

		self.cache['draw'] = np.concatenate((I_o,flegend, fplane, cbar, nbar, self.cache['draw']),axis=0)

		displayThread.acquire()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		displayThread.release()

		self.t.append(time.time()-self.t0)

	def iccp_output(self):	
		# backup the data for saving
		self.depth_data['Z'] = copy.deepcopy(self.results['Z'])
		self.depth_data['conf'] = copy.deepcopy(self.results['conf'])

		if self.to_show():
			conf_thre = 0.999
		else:
			conf_thre = 1

		Z = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			conf_thre
		)
		
		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.I_cache[:,:,1].astype(np.float32)
		self.results['I_1'] = self.I_cache[:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		# create the image to draw
		Z = self.prep_for_draw_demo(I = Z, message='Depth', rng=KEY_RANGE['raw'])
		I = self.prep_for_draw_demo(I = self.results['I_0'], message='Input image', rng=KEY_RANGE['raw'])
		self.cache['draw'] = np.concatenate((I,Z), axis=1)

		# new shape
		ncol = I.shape[1] + Z.shape[1]

		# in-focus plane
		flegend = np.zeros((20, ncol), dtype=np.float32)
		fplane = np.zeros((10, ncol), dtype=np.float32)
		Z_f = self.cfg[0]['a1']
		print(Z_f)
		loc = int((Z_f + DEPTH_RANGE[1])/(DEPTH_RANGE[1]-DEPTH_RANGE[0])*ncol)
		fplane[:, loc-2:loc+2] = 1
		fplane = self.prep_for_draw_demo(I = fplane, rng=[0,1])
		flegend = self.prep_for_draw_demo(I = flegend, rng=[0,1])
		
		t_s = 0.5
		t_h = int(20*t_s)
		flegend = cv2.putText(\
			flegend, \
			"Mean Depth", \
			(loc-45,t_h+3), \
			FONT, \
			t_s, \
			(1,1,1)\
		)

		# make a color bar
		cbar_row = 10
		seg = 100
		# color bar
		cbar = np.matlib.repmat(np.linspace(100,0,num=seg), cbar_row, 1)
		# pseudo confidence
		cpseudo = np.ones((cbar_row, seg))
		# give the pseudo color
		cbar = (self.pseudo_color_Z(cbar, cpseudo, 0, 100, 0) * 255).astype(np.uint8)
		cbar = scipy.misc.imresize(cbar, (cbar_row, ncol)).astype(np.float32)/255

		nums = np.linspace(0.3,1.0,8)
		for i in range(nums.shape[0]-1):
			loc = int(ncol / (nums.shape[0]-1) * i)
			cbar[-6::,loc,0] = 0
			cbar[-6::,loc,1] = 0
			cbar[-6::,loc,2] = 0

		# make the number bar
		nbar = np.zeros((30, cbar.shape[1]),dtype=np.float32)
		nbar = self.prep_for_draw_demo(I = nbar, rng=[0,100])
		t_s = 0.5
		t_h = int(20*t_s)
		nbar = cv2.putText(\
			nbar, \
			"Depth(m)",\
			(0, t_h+5), \
			FONT, \
			t_s, \
			(1,1,1)\
		)
		for i in range(nums.shape[0]-2):
			loc = int(ncol / (nums.shape[0]-1) * (i+1))
			nbar = cv2.putText(\
				nbar, \
				str(nums[i+1]), \
				(loc-13, t_h+5), \
				FONT, \
				t_s, \
				(1,1,1)\
			)

		self.cache['draw'] = np.concatenate((flegend, fplane, cbar, nbar, self.cache['draw']),axis=0)

		displayThread.acquire()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		displayThread.release()

		self.t.append(time.time()-self.t0)

	def bilateral(self, I, Z, conf):
		return cv2.bilateralFilter(Z,5,0.3,3)

	import cv2.ximgproc
	def guided(self, I, Z, conf):
		return cv2.ximgproc.guidedFilter(I, Z, 9, 0.3)

	def jointBilateral(self, I, Z, conf):
		return cv2.ximgproc.jointBilateralFilter(\
			I,\
			Z,\
			9,\
			5,\
			7,
		)

	import scipy.signal
	def iccv_output(self):	
		# backup the data for saving
		self.depth_data['Zf'] = copy.deepcopy(self.results['Zf'])
		self.depth_data['conff'] = copy.deepcopy(self.results['conf'])

		if self.to_show():
			conf_thre = 0
		else:
			conf_thre = 0

		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.I_cache[:,:,1].astype(np.float32)
		self.results['I_1'] = self.I_cache[:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		# # cut out those who are saturated
		# flg = np.where(np.maximum(self.results['I_0'],self.results['I_1'])>= 250)
		# self.results['conf'][flg] = 0
		# tmp = np.ones(self.results['conf'].shape)
		# tmp[np.where(np.isnan(self.results['conf']))] = np.nan
		# cut_edge_filt = np.ones([10,10])
		# tmp = scipy.signal.convolve2d(tmp,cut_edge_filt,mode='same')
		# self.results['conf'][np.where(np.isnan(tmp))] = 0
		# self.results['conf'][np.where(np.isnan(self.results['conf']))] = 0

		# flag = np.zeros(self.results['conf'].shape)
		# self.results['conf'] = self.results['conf'] * 

		# # cut out all parts that is below confidence
		# filter_thre = 0.9
		# self.results['conf'][np.where(self.results['conf']<filter_thre)] = 0.001
		# self.results['Zf'] = self.guided(\
		# 	self.results['I_0'],\
		# 	self.results['Zf'], \
		# 	self.results['conf']\
		# )

		# cut out things that are not correct
		# too far away from Z_f
		if self.show_modes[self.show_mode] == 'origin':
			Z_f = self.cfgf[0]['a1']
			flag = np.where(\
					(np.abs(self.results['Zf']) - np.abs(Z_f)) > 0.5*Z_f
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			# too close than Z_f
			flag = np.where(\
					(np.abs(self.results['Zf']) - np.abs(Z_f)) < -0.5*Z_f
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			# confidence smaller than the threshold
			flag = np.where(\
					self.results['conf']<0.999
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10


			# saturated region
			I = (self.results['I_0']+self.results['I_1'])/2
			flag = np.where(\
					I > 250
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			self.results['Zf'] = self.bilateral(\
				self.results['I_0'],\
				self.results['Zf'], \
				self.results['conf']\
			)
		elif self.show_modes[self.show_mode] == 'partial BFS':
			Z_f = self.cfgf[0]['a1']
			flag = np.where(\
					(np.abs(self.results['Zf']) - np.abs(Z_f)) > 0.5*Z_f
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			# too close than Z_f
			flag = np.where(\
					(np.abs(self.results['Zf']) - np.abs(Z_f)) < -0.5*Z_f
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			# confidence smaller than the threshold
			flag = np.where(\
					self.results['conf']<0.999
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10


			# saturated region
			I = (self.results['I_0']+self.results['I_1'])/2
			flag = np.where(\
					I > 250
				)
			self.results['Zf'][flag] = -1e10
			self.results['conf'][flag] = -1e10

			# try the fast bilateral solver
			grid_params = {
			    'sigma_luma' : 10,
			    'sigma_chroma': 10,
			    'sigma_spatial': 3
			}

			I = np.stack([self.results['I_0'] for i in range(3)], -1)
			Z = self.results['Zf']
			C = self.results['conf']
			C = 1/(1-C)
			grid = BilateralGrid(I, **grid_params)

			t = Z.reshape(-1, 1).astype(np.float32) #/ (pow(2,16)-1)
			c = C.reshape(-1, 1).astype(np.float32) #/ (pow(2,16)-1)
			tc_filt = grid.filter(t * c)
			c_filt = grid.filter(c)
			output = (tc_filt / c_filt).reshape(self.results['Zf'].shape)
			self.results['Zf'] = output

			t = C.reshape(-1, 1).astype(np.float32)
			tc_filt = grid.filter(t * c)
			output = (tc_filt / c_filt).reshape(self.results['conf'].shape)
			output = 1-1/output
			self.results['conf'] = output
		elif self.show_modes[self.show_mode] == 'full BFS':
			# try the fast bilateral solver
			grid_params = {
			    'sigma_luma' : 10,
			    'sigma_chroma': 10,
			    'sigma_spatial': 3
			}

			I = np.stack([self.results['I_0'] for i in range(3)], -1)
			Z = self.results['Zf']
			C = self.results['conf']
			C = 1/(1-C)
			grid = BilateralGrid(I, **grid_params)

			t = Z.reshape(-1, 1).astype(np.float32) #/ (pow(2,16)-1)
			c = C.reshape(-1, 1).astype(np.float32) #/ (pow(2,16)-1)
			tc_filt = grid.filter(t * c)
			c_filt = grid.filter(c)
			output = (tc_filt / c_filt).reshape(self.results['Zf'].shape)
			self.results['Zf'] = output

		# # clustering 
		# Zf = np.reshape(self.results['Zf'][np.where(self.results['Zf']> -1e10)],(-1,1))
		# pdb.set_trace()
		# xf = self.clustering.fit(Zf)
		# pdb.set_trace()


		# self.results['Zf'] = self.bilateral(\
		# 	self.results['I_0'],\
		# 	self.results['Zf'], \
		# 	self.results['conf']\
		# )

		

		# swap the left and right
		self.results['Zf'] = self.results['Zf'][:,::-1]
		self.results['I_0'] = self.results['I_0'][:,::-1]
		self.results['conf'] = self.results['conf'][:,::-1]

		Z = self.pseudo_color_Z(\
			self.results['Zf'],\
			self.results['conf'],\
			DEPTH_RANGE_f[0],\
			DEPTH_RANGE_f[1],\
			conf_thre
		)

		# create the image to draw
		# Z = self.prep_for_draw_demo(I = self.results['I_0'], message='Depth', rng=KEY_RANGE['raw'])
		# I = self.prep_for_draw_demo(I = np.abs(self.results['u_2']/self.results['I_0'])*500, message='Input image', rng=KEY_RANGE['raw'])
		Z = self.prep_for_draw_demo(I = Z, message='Depth', rng=KEY_RANGE['raw'])
		I = self.prep_for_draw_demo(I = self.results['I_0'], message='Input image', rng=KEY_RANGE['raw'])
		
		self.cache['draw'] = np.concatenate((I,Z), axis=1)

		# new shape
		ncol = I.shape[1] + Z.shape[1]

		# in-focus plane
		flegend = np.zeros((20, ncol), dtype=np.float32)
		fplane = np.zeros((10, ncol), dtype=np.float32)
		Z_f = self.cfgf[0]['a1']
		# print(Z_f)
		loc = int((Z_f + DEPTH_RANGE_f[1])/(DEPTH_RANGE_f[1]-DEPTH_RANGE_f[0])*ncol)
		loc = np.maximum(loc, 0)
		fplane[:, loc-2:loc+2] = 1
		fplane = self.prep_for_draw_demo(I = fplane, rng=[0,1])
		flegend = self.prep_for_draw_demo(I = flegend, rng=[0,1])
		
		t_s = 0.5
		t_h = int(20*t_s)
		flegend = cv2.putText(\
			flegend, \
			"Mean Depth", \
			(loc-45,t_h+3), \
			FONT, \
			t_s, \
			(1,1,1)\
		)

		# make a color bar
		cbar_row = 10
		seg = 100
		# color bar
		cbar = np.matlib.repmat(np.linspace(100,0,num=seg), cbar_row, 1)
		# pseudo confidence
		cpseudo = np.ones((cbar_row, seg))
		# give the pseudo color
		cbar = (self.pseudo_color_Z(cbar, cpseudo, 0, 100, 0) * 255).astype(np.uint8)
		cbar = scipy.misc.imresize(cbar, (cbar_row, ncol)).astype(np.float32)/255

		nums = np.linspace(-DEPTH_RANGE_f[1],-DEPTH_RANGE_f[0],9)
		for i in range(nums.shape[0]-1):
			loc = int(ncol / (nums.shape[0]-1) * i)
			cbar[-6::,loc,0] = 0
			cbar[-6::,loc,1] = 0
			cbar[-6::,loc,2] = 0

		# make the number bar
		nbar = np.zeros((30, cbar.shape[1]),dtype=np.float32)
		nbar = self.prep_for_draw_demo(I = nbar, rng=[0,100])
		t_s = 0.5
		t_h = int(20*t_s)
		nbar = cv2.putText(\
			nbar, \
			"Depth(m)",\
			(0, t_h+5), \
			FONT, \
			t_s, \
			(1,1,1)\
		)
		for i in range(nums.shape[0]-2):
			loc = int(ncol / (nums.shape[0]-1) * (i+1))
			nbar = cv2.putText(\
				nbar, \
				str(nums[i+1]), \
				(loc-13, t_h+5), \
				FONT, \
				t_s, \
				(1,1,1)\
			)

		# # outside color 
		# I_o = scipy.misc.imresize(self.outside_I, \
		# 	(int(ncol/self.outside_I.shape[1]*self.outside_I.shape[0]), ncol))
		# I_o = self.prep_for_draw_demo(I = I_o, rng=KEY_RANGE['raw'])

		# self.cache['draw'] = np.concatenate((I_o,flegend, fplane, cbar, nbar, self.cache['draw']),axis=0)

		self.cache['draw'] = np.concatenate((flegend, fplane, cbar, nbar, self.cache['draw']),axis=0)

		displayThread.acquire()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		displayThread.release()

		self.t.append(time.time()-self.t0)

	def regular_output(self):
		global DEPTH_RANGE
		# backup the data for saving

		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		self.results['Z'] = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			0.999
		)

		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.I_cache[:,:,0].astype(np.float32)
		self.results['I_1'] = self.I_cache[:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		rng = {}
		self.image_to_show = ['I_0','Z']
		tmp = {}
		for k in self.image_to_show:
			if k in KEY_RANGE.keys():
				rng[k] = KEY_RANGE[k]
			else:
				rng[k] = [np.NaN, np.NaN]
			tmp[k] = self.results[k]

		rng['I_0'] = KEY_RANGE['raw']
		rng['I_1'] = KEY_RANGE['raw']

		self.cache['draw'] = tile_image(\
									I = tmp, \
									rng = rng, \
									log = False, \
									title = "Regular Output", \
								)
		# self.save_video()

		displayThread.acquire()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		displayThread.release()
		
		self.t.append(time.time()-self.t0)

	def regular_output_pairs(self):
		global DEPTH_RANGE
		# backup the data for saving

		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		self.results['Z'] = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			0.999
		)
		self.results['Zf'] = self.pseudo_color_Z(\
			self.results['Zf'],\
			self.results['conff'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			0.999
		)

		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.I_cache[:,:,0].astype(np.float32)
		self.results['I_1'] = self.I_cache[:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		rng = {}
		self.image_to_show = ['I_0','I_1','Z','Zf']
		tmp = {}
		for k in self.image_to_show:
			if k in KEY_RANGE.keys():
				rng[k] = KEY_RANGE[k]
			else:
				rng[k] = [np.NaN, np.NaN]
			tmp[k] = self.results[k]

		rng['I_0'] = KEY_RANGE['raw']
		rng['I_1'] = KEY_RANGE['raw']

		self.cache['draw'] = tile_image(\
									I = tmp, \
									rng = rng, \
									log = False, \
									title = "Regular Output", \
								)
		# self.save_video()

		displayThread.acquire()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		displayThread.release()
		
		self.t.append(time.time()-self.t0)

	def regular_output_layers(self):
		# query data
		self.image_to_show = ['Z','conf']
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars_fuse[k]

		res_dict['Z_align'] = self.vars_align['Z']
		res_dict['conf_align'] = self.vars_align['conf']

		self.results = self.session.run(res_dict)	
		
		# backup the data for saving
		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		conf_thre = 0.995
		self.results['Z'] = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			conf_thre
		)
		# print out the Z in each layer
		for i in range(len(self.cfg)):
			for j in range(self.cfg[0]['ext_f'].shape[0]):
				idx = i * (self.cfg[0]['ext_f'].shape[0]) + j
				self.results['Z'+str(idx)] = self.results['Z_align'][:,:,idx]
				self.results['conf'+str(idx)] = self.results['conf_align'][:,:,idx]

				# back up the data for saving
				self.depth_data['Z'+str(idx)] = self.results['Z'+str(idx)]
				self.depth_data['conf'+str(idx)] = self.results['conf'+str(idx)]


				self.results['Z'+str(idx)] = self.pseudo_color_Z(\
					self.results['Z'+str(idx)],\
					self.results['conf'+str(idx)],\
					DEPTH_RANGE[0],\
					DEPTH_RANGE[1],\
					conf_thre
				)
				self.image_to_show.append('Z'+str(idx))
				self.image_to_show.append('conf'+str(idx))

		del self.results['Z_align']
		del self.results['conf_align']


		self.results['I_0'] = self.input_dict[self.I_in][:,:,0].astype(np.float32)
		self.results['I_1'] = self.input_dict[self.I_in][:,:,1].astype(np.float32)

		# backup the data for saving
		self.depth_data['I_0'] = self.results['I_0']
		self.depth_data['I_1'] = self.results['I_1']

		self.valid_region_I()

		rng = {}
		for k in self.image_to_show:
			if k in KEY_RANGE.keys():
				rng[k] = KEY_RANGE[k]
			else:
				rng[k] = [np.NaN, np.NaN]
		rng['I_0'] = KEY_RANGE['raw']
		rng['I_1'] = KEY_RANGE['raw']

		self.cache['draw'] = tile_image(\
									I = self.results, \
									rng = rng, \
									log = False, \
									title = "Regular Output", \
								)
		# self.save_video()
		cv2.imshow("Focal Track Demo", self.cache['draw'])
		self.t.append(time.time()-self.t0)

	def prep_for_draw_demo(self, I, log = False, title = None, message = None, rng = [np.NaN, np.NaN]):
		if len(I.shape) == 2 or I.shape[2] == 1:
			valid = np.isfinite(I)
			invalid = ~valid
			
			if (not np.all(invalid)) and (not np.all(~np.isnan(rng))):
				# pdb.set_trace()
				rng = [np.min(I[valid]), np.max(I[valid])]

			#convert to color image
			D = I.copy()
			
			D = np.float32(D)

			if len(I.shape) == 2 or I.shape[2] == 1:
				D = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
			if D.shape[2] != 3:
				raise Exception("Unsupported shape for prepping for draw: " + str(D.shape))	
			
			D = (D-rng[0])/float(rng[1] - rng[0])
			
			if log:
				D = np.log10(D)
				D_rng = [np.min(D), np.max(D)]
				D = (D-D_rng[0])/float(D_rng[1] - D_rng[0])
			
			#D = np.uint8(np.clip(255.0*D, 0, 255))
			if invalid.any():
				D[invalid] = NAN_COLOR
		else:
			D = I
		
		# D[I > rng[1]] = NAN_COLOR
		# D[I < rng[0]] = NAN_COLOR
		
		t_s = 0.7
		t_h = int(20*t_s)
		title_str = "[%1.1e,%2.1e]"%(rng[0], rng[1])
		if title is not None:
			title_str = title  + "::" + title_str
		
		if message is not None:
			#message = "fps: %2.2f"%self.get_fps()
			cv2.putText(D, message, (5, t_h+5), FONT, t_s, (1,1,1))
		return D

	def pseudo_color_Z(self, Z, conf, lo, hi, conf_thre):
		# cut out the region
		Z[np.where(conf <= conf_thre)] = lo
		Z[np.where(Z<lo)] = lo
		Z[np.where(Z>hi)] = hi

		# Z[np.where(conf <= conf_thre)] = -1e10
		# Z[np.where(Z<lo)] = -1e10
		# Z[np.where(Z>hi)] = -1e10

		# convert to pseudo color
		Z_g = (Z-lo)/(hi-lo)*255
		Z_g = Z_g.astype(np.uint8)
		Z_rgb = cv2.applyColorMap(Z_g, cv2.COLORMAP_JET)

		idx = np.where(conf <= conf_thre) or\
			np.where(Z <= lo) or\
			np.where(Z >= hi)

		# pdb.set_trace()

		Z_rgb[:,:,0][idx] = 0
		Z_rgb[:,:,1][idx] = 0
		Z_rgb[:,:,2][idx] = 0


		return Z_rgb.astype(np.float32)/255

	def save_video(self):
		if self.video_writer == None:
			fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
			directory = "./test_results/pyConfLensFlowNetFast/"
			lavi = len(glob.glob(\
				directory + "*.avi"
			))
			# watch out, we should reverse the cols and rows shape here
			self.video_writer = cv2.VideoWriter(\
				"./test_results/pyConfLensFlowNetFast/"+str(lavi)+".avi",\
				fourcc, 20.0, (self.cache['draw'].shape[1],self.cache['draw'].shape[0])
			)
		# tmp = (self.cache['draw'][:,:,0],self.cache['draw'][:,:,1],self.cache['draw'][:,:,2])
		# tmp = np.concatenate(tmp, axis = 0)
		tmp = np.transpose(self.cache['draw'], axes = (2,1,0))
		tmp = np.reshape(tmp, -1)
		tmp = np.transpose(np.reshape(tmp, (self.cache['draw'].shape[1],self.cache['draw'].shape[0]*3)))
		self.video_writer.write(tmp)
		return 

	def valid_region_I(self):
		vars_to_cut = [\
			'u_2','I_0','I_1'\
		]

		rows_cut = int(\
			((self.cfg[-1]['gauss'].shape[0]-1)/2+\
			(self.cfg[-1]['ext_f'].shape[1]-1)/2)*\
			self.resolution[0][0]/self.resolution[-1][0]
		)
		cols_cut = int(\
			((self.cfg[-1]['gauss'].shape[1]-1)/2+\
			(self.cfg[-1]['ext_f'].shape[2]-1)/2)*\
			self.resolution[0][1]/self.resolution[-1][1]
		)

		rows = self.cfg[0]['szx_sensor']
		cols = self.cfg[0]['szy_sensor']

		for var in vars_to_cut:
			self.results[var] = \
				self.results[var][
					cols_cut:cols-cols_cut,
					rows_cut:rows-rows_cut
				]

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
			####################################
			### DISPLAY PROCESSING FPS STATISTICS ###
			####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})

	"""destructor: free up resources when done"""
	def __del__(self):
		# self.video_writer.release()
		cv2.destroyAllWindows()


def multithreading_test():
	# c = OutsideCamera()
	a = Camera()

	# c.start()
	a.start()

	time.sleep(1)

	# initialize the pulsecam processor
	# cfg_file = "./opt_results/pyConfLensFlowNetFast/"+\
	# 	"1x1t-text34-setup5-py4-w3r-whole.pickle"
	cfg_file = "./extra/params/"+\
		"1x1t-text34-py4-setup5-one-sequential-regularize-nothreshold.pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']

	for i in range(len(cfg)):
		cfg[i]['b0'] = 1e-4
		cfg[i]['b1'] = -1.9

	cfg_file = "./extra/params/"+\
		"final.pickle"
	with open(cfg_file,'rb') as f:
		cfgf_data = pickle.load(f)
	cfgf = cfgf_data['cfg']

	for i in range(len(cfgf)):
		cfgf[i]['a1'] = 0
		cfgf[i]['ra1_1'] = 0
		cfgf[i]['ra1_2'] = 0

	b = PulseCamProcessorTF(cfg[0:-1], cfgf)
	b.start()

	time.sleep(1)
	d = Display(cfg[0:-1], cfgf)	
	d.start()
	
	# c.join()
	a.join()

	b.join()
	d.join()

if __name__ == "__main__":
	# debug_test()
	multithreading_test()
