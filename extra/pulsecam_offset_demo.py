#!/usr/bin/python2.7
import serial
import argparse

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
from training_pyConfLensFlowNetFast import training_pyConfLensFlowNetFast

from utils import *
import time

#notes
#  -did I mix up x and y again?  always worth checking

#debugging
import pdb

#multithreading
import threading
myThread = threading.Condition()
outsideThread = threading.Condition()
I_cache = 0
outside_I = 0
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
			# self.regular_output()
			c = cv2.waitKey(1) & 0xFF
			if chr(c).lower() == 'q' or ending_key == 'q':
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("quitting")
				print("outside camera frame rate: (gui + compute)", perf, " fps")
				self.final_statistics()
				ending_key = 'q'
				break

			self.t.append(time.time()-self.t0)

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
		outsideThread.acquire()
		outside_I = self.outside_I
		outsideThread.release()

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
		self.draw = None
		self.cache = {}
		self.vars = {}
		self.frames = 0
		self.resolution = None
		self.counter = 0

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

	def run(self):
		global ending_key
		# The code to capture images
		t0 = time.time()
		while True:
			self.t0 = time.time()
			self.grab_frame_and_process_ptg()

			self.regular_output()

			c = cv2.waitKey(1) & 0xFF
			if chr(c).lower() == 'q' or ending_key == 'q':
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("quitting")
				print("camera capture frame rate: (gui + compute)", perf, " fps")
				self.final_statistics()
				ending_key = 'q'
				break
			if chr(c).lower() == 's':
				# switch the two images to always make the one
				# with smaller P (larger f) in front
				self.idx = 1 - self.idx
				tmp = self.I_cache[:,:,0]
				self.I_cache[:,:,0] = self.I_cache[:,:,1]
				self.I_cache[:,:,1] = tmp

			self.t.append(time.time()-self.t0)
		
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
		
		# decide which idx of image should be updated
		# if self.counter > 2:
		# 	self.decide_idx(self.cache['gray'])
		# else:
		# 	self.naive_idx()
		self.naive_idx()
		self.counter += 1

		self.I_cache[:,:,self.idx] = self.cache['gray']

		# Lock the global variables, to updates those images
		myThread.acquire()
		I_cache = self.I_cache
		myThread.release()

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

class PulseCamProcessor(threading.Thread, training_pyConfLensFlowNetFast):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg):
		threading.Thread.__init__(self)
		
		self.cfg = cfg
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
		
		self.required_outs = ['Z']
		self.vars_to_fuse = ['Z','conf']
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

		
		#initialize computation
		self.Z_t = []
		
		self.results = {}

		# the port
		self.Z_tgt = self.cfg[0]['a1']
		self.Z_tgt0 = self.cfg[0]['a1']
		for i in range(len(self.cfg)):
			self.cfg[i]['a0a1'] = self.cfg[i]['a0'] * self.cfg[0]['a1']

		self.offset_range = [9000,56535]
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
		self.track_methods = ['passive_Z', 'track_Z_pid']
		self.track_idx = 0


		self.sample_images = None
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)

		self.data_type = tf.float32

		self.I_cache = np.zeros(self.resolution[0]+(2,), dtype = np.uint8)
		self.outside_I = 0
		self.depth_data = {}

		# make a video recorder
		self.video_writer = None
		self.build_graph()

	def run(self):
		global ending_key
		t0 = time.time()
		while True:
			self.t0 = time.time()
			self.process()
			self.demo_output()
			eval("self."+self.track_methods[self.track_idx]+"()")

			c = cv2.waitKey(1) & 0xFF
			if chr(c).lower() == 'q' or ending_key == 'q':
				t1 = time.time()
				perf = (1.0*self.frames)/(t1-t0)
				print("quitting")
				print("FF avg performance: (gui + compute)", perf, " fps")
				self.final_statistics()
				ending_key = 'q'
				break
			if chr(c) == '\n':
				# collect the selected data
				directory = "./models/"
				lavi = len(glob.glob(\
					directory + "*.pickle"
				))
				fileName = "./models/"\
					+ str(lavi) + ".pickle"
				with open(fileName,'wb') as f:
					pickle.dump(self.depth_data, f)

			if chr(c).lower() == 't':
				scipy.misc.imsave('./models/demoshot/plain.png', \
					(self.cache['draw']*255).astype(np.uint8)
				)
			if chr(c).lower() == 'r':
				# change tracking strategy
				self.track_idx += 1
				if self.track_idx == len(self.track_methods):
					self.track_idx = 0


			self.frames += 1
			self.t.append(time.time()-self.t0)

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""

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

			I_batch = []
			I_lap_batch = []

			Z_0 = []
			a0a1 = []
			a0 = []
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

			# depth computation
			# I used to be (960, 600, 2or 3) we change it to (2or3, 960, 600) 
			tmp_I = tf.transpose(I, perm=[2,0,1])
			for i in range(len(self.cfg)):
				# initialize variables				
				"""Input parameters"""
				Z_0.append(tf.constant(self.cfg[i]['Z_0'], dtype = tf.float32))
				a0a1.append(tf.constant(self.cfg[i]['a0a1'], dtype = tf.float32))
				a0.append(a0a1[i]/a1)

				gauss.append(tf.Variable(self.cfg[i]['gauss'], dtype = tf.float32))
				ext_f.append(tf.Variable(self.cfg[i]['ext_f'], dtype = tf.float32))
				ft.append(tf.constant(self.cfg[i]['ft'], dtype = tf.float32))
				fave.append(tf.constant(self.cfg[i]['fave'], dtype = tf.float32))

				I_batch.append(tf.transpose(tmp_I,[1,2,0]))
				tmp_I_blur = dIdx_batch(tmp_I, gauss[i])
				I_lap_batch.append(tf.transpose(tmp_I - tmp_I_blur,[1,2,0]))

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

				# Generate the differential images
				I_t.append(dIdt(I_batch[i], ft[i]))
				I_lap.append(dIdt(I_lap_batch[i], fave[i]))

				# unwindowed version
				u_1.append(I_lap[i])
				u_2.append(I_t[i])
				u_3.append(a0[i] * a1 * u_1[i])
				u_4.append(-u_2[i] + a0[i] * u_1[i] + 1e-5)
				ext_f[i] = tf.expand_dims(\
					tf.transpose(ext_f[i], perm=[1,2,0]),-2
				)
				u_3[i] = tf.expand_dims(tf.expand_dims(u_3[i],0),-1)
				u_4[i] = tf.expand_dims(tf.expand_dims(u_4[i],0),-1)
				u_3[i] = tf.nn.conv2d(u_3[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_4[i] = tf.nn.conv2d(u_4[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]

				
				#save references to required I/O]
				self.vars.append({})
				self.vars[i]['I'] = I_batch[i]
				
				# unwindowed version
				self.vars[i]['u_1'] = u_1[i]
				self.vars[i]['u_2'] = u_2[i]
				self.vars[i]['u_3'] = u_3[i]
				self.vars[i]['u_4'] = u_4[i]

			# align depth and confidence maps
			self.align_maps_ext(['u_3','u_4'])

			# compute the aligned version of Z
			self.vars_align['Z'] = \
				self.vars_align['u_3'] / (self.vars_align['u_4'] + 1e-5) + Z_0[0]

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[0]['conf_func']+'()')

			# fusion
			self.softmax_fusion()

			# 
			self.valid_windowed_region_fuse()

			self.vars_fuse['Z'] -= Z_0[0]
			self.vars_fuse['Z'] = -self.vars_fuse['Z']

			self.vars_align['Z'] -= Z_0[0]
			self.vars_align['Z'] = -self.vars_align['Z']

			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			self.input_data = tf.group(\
				I.assign(self.I_in), a1.assign(self.a1_in)
			)

			#do not add anything to the compute graph 
			#after this line
			init_op = tf.initialize_all_variables()
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
				tf.concat(3, self.vars_align[var]), [0]
			)
		return 

	def softmax_fusion(self):
		# reshape for softmax
		conf_flat = tf.reshape(
			self.vars_align['conf'],
			[-1, len(self.cfg)*self.cfg[0]['ext_f'].shape[0]]
		)
		
		# not sure if it will cause numerical problem
		ws = tf.reshape(
			tf.nn.softmax(conf_flat*1e10),
			self.resolution[0]+(-1,)
		)

		# fuse the results using softmax
		for var in self.vars_to_fuse:
			self.vars_fuse[var] = \
				tf.reduce_sum(
					self.vars_align[var]*ws,
					[2]
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
		k = 50
		for i in range(len(self.cfg)):
			# weights
			w_bc.append(\
				tf.Variable(
					self.cfg[i]['w_bc'],
					dtype = tf.float32
				)	
			)
			w_bc1.append(\
				tf.Variable(
					self.cfg[i]['w_bc1'],
					dtype = tf.float32
				)	
			)
			w_bc2.append(\
				tf.Variable(
					self.cfg[i]['w_bc2'],
					dtype = tf.float32
				)
			)

		# aligned version
		# self.align_maps(['conf'])
		conf_align = []

		for i in range(len(self.cfg)):
			for j in range(self.cfg[0]['ext_f'].shape[0]):
				idx = i * (self.cfg[0]['ext_f'].shape[0]) + j
				tmp_align = (self.vars_align['u_4'][:,:,idx]**2 + 1e-20)/\
					tf.sqrt(\
						w_bc[i][j] * self.vars_align['u_3'][:,:,idx]**2 + \
						w_bc1[i][j] * self.vars_align['u_4'][:,:,idx]**2 + \
						w_bc2[i][j] * self.vars_align['u_3'][:,:,idx]*self.vars_align['u_4'][:,:,idx] + \
						self.vars_align['u_4'][:,:,idx]**4 + \
						1e-10\
					)	
				# aligned confidence
				conf_align.append(
					tmp_align
				)
			self.vars[i]['conf_align'] = conf_align[i]

		self.vars_align['conf'] = tf.pack(conf_align,2)

		return

	def valid_windowed_region_fuse(self):
		# cut out the bad part
		vars_to_cut = [\
			'Z','conf'\
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
		global I_cache
		global outside_I

		myThread.acquire()
		self.input_dict[self.I_in] = I_cache
		myThread.release()

		outsideThread.acquire()
		self.outside_I = outside_I
		outsideThread.release()

		self.input_dict[self.a1_in] = self.cfg[0]['a1']

		self.session.run(self.input_data, self.input_dict)
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
			####################################
			### PC PROCESSING FPS STATISTICS ###
			####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})

	def passive_Z(self):
		self.Z_tgt = np.abs(np.median(self.depth_data['Z'][np.where(self.depth_data['conf'] > 0.999)]))
		if np.isnan(self.Z_tgt):
			self.Z_tgt = self.Z_tgt0

		self.offset = int((1/self.Z_tgt + 1.9)*1e4)
		if self.offset < self.offset_range[0]:
			self.offset = self.offset_range[0]
		elif self.offset > self.offset_range[1]:
			self.offset = self.offset_range[1]
		# print("Z_target", self.Z_tgt, "offset: ", self.offset)
		return

	def track_Z_pid(self):
		global DEPTH_RANGE
		Z_tmp = self.depth_data['Z'][:,int(self.depth_data['Z'].shape[1]/2):-1]
		conf_tmp = self.depth_data['conf'][:,int(self.depth_data['conf'].shape[1]/2):-1]
		# Z_tmp = self.depth_data['Z']
		# conf_tmp = self.depth_data['conf']
		self.Z_tgt = np.abs(np.median(Z_tmp[np.where(conf_tmp > 0.999)]))
		err = self.cfg[0]['a1'] - self.Z_tgt
		self.inte += err
		der = err - self.prev_err 

		# Kp = -0.05
		# Ki = -0.001
		# Kd = -0.01

		Kp = -0.2
		Ki = -0.0001
		Kd = -0.01

		dZ = Kp*err + Ki*self.inte + Kd*der
		print("err:",err,"inte:",self.inte,"der",der,"dZ",dZ)
		Z_set = self.cfg[0]['a1'] +dZ
		self.prev_err = err

		if np.isnan(Z_set):
			Z_set = self.Z_tgt0
			self.inte = 0
			self.prev_err = 0

		self.offset = int((1/Z_set + 1.9)*1e4)

		if self.offset < self.offset_range[0]:
			self.offset = self.offset_range[0]
		elif self.offset > self.offset_range[1]:
			self.offset = self.offset_range[1]
		

		self.cfg[0]['a1'] = 1/(1e-4*self.offset-1.9)
		# DEPTH_RANGE = [-self.cfg[0]['a1']-0.1,-self.cfg[0]['a1']+0.1]
		KEY_RANGE['Z'] = DEPTH_RANGE
		print("Z_target: ", self.Z_tgt, "Z_set: ", Z_set,"a1: ",self.cfg[0]['a1'], "offset: ", self.offset)
		self.control_lens()
		return

	def track_Z(self):
		self.Z_tgt = np.abs(np.median(self.depth_data['Z'][np.where(self.depth_data['conf'] > 0.999)]))
		if np.isnan(self.Z_tgt):
			self.Z_tgt = self.Z_tgt0

		self.offset = int((1/self.Z_tgt + 1.9)*1e4)
		if self.offset < self.offset_range[0]:
			self.offset = self.offset_range[0]
		elif self.offset > self.offset_range[1]:
			self.offset = self.offset_range[1]
		# print("Z_target", self.Z_tgt, "offset: ", self.offset)

		self.cfg[0]['a1'] = 1/(1e-4*self.offset-1.9)
		self.control_lens()
		return

	def control_lens(self):
		if self.ser.isOpen():
			string = "OF"+str(self.offset)+";"
			self.ser.write(string.encode())
			response = self.ser.read(self.ser.inWaiting())
			print(response)
		return

	def demo_output(self):
		global DEPTH_RANGE
		self.image_to_show = ['Z','conf']
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars_fuse[k]

		self.results = self.session.run(res_dict)
		
		# backup the data for saving
		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		Z = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			0.995
		)
		# print("DEPTH_RANGE", DEPTH_RANGE)
		self.results['I_0'] = self.input_dict[self.I_in][:,:,0].astype(np.float32)
		self.results['I_1'] = self.input_dict[self.I_in][:,:,1].astype(np.float32)

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
		I_o = scipy.misc.imresize(outside_I, (int(ncol/outside_I.shape[1]*outside_I.shape[0]), ncol))
		I_o = self.prep_for_draw_demo(I = I_o, rng=KEY_RANGE['raw'])

		self.cache['draw'] = np.concatenate((I_o,flegend, fplane, cbar, nbar, self.cache['draw']),axis=0)

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

	def regular_output(self):
		global DEPTH_RANGE
		self.image_to_show = ['Z','conf']
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars_fuse[k]

		self.results = self.session.run(res_dict)
		
		# backup the data for saving
		self.depth_data['Z'] = self.results['Z']
		self.depth_data['conf'] = self.results['conf']

		self.results['Z'] = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			0.995
		)
		# print("DEPTH_RANGE", DEPTH_RANGE)
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
		cv2.imshow("PulseCam Thread", self.cache['draw'])
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
		cv2.imshow("PulseCam Thread", self.cache['draw'])
		self.t.append(time.time()-self.t0)

	def pseudo_color_Z(self, Z, conf, lo, hi, conf_thre):
		# cut out the region
		Z[np.where(conf <= conf_thre)] = lo
		Z[np.where(Z<lo)] = lo
		Z[np.where(Z>hi)] = hi

		# convert to pseudo color
		Z_g = (Z-lo)/(hi-lo)*255
		Z_g = Z_g.astype(np.uint8)
		Z_rgb = cv2.applyColorMap(Z_g, cv2.COLORMAP_JET)

		idx = np.where(conf <= conf_thre) or\
			np.where(Z <= lo) or\
			np.where(Z >= hi)

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
			'I_0','I_1'\
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
			
	"""destructor: free up resources when done"""
	def __del__(self):
		# self.video_writer.release()
		cv2.destroyAllWindows()


def multithreading_test():
	c = OutsideCamera()
	a = Camera()

	c.start()
	a.start()

	time.sleep(1)

	# initialize the pulsecam processor
	# cfg_file = "./opt_results/pyConfLensFlowNetFast/"+\
	# 	"1x1t-text34-setup5-py4-w3r-whole.pickle"
	cfg_file = "./opt_results/pyConfLensFlowNetFast_ext/"+\
		"1x1t-text34-py4-setup5-one-sequential-regularize-nothreshold.pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']
	
	b = PulseCamProcessor(cfg[0:-1])

	b.start()
	
	c.join()
	a.join()

	b.join()

if __name__ == "__main__":
	# debug_test()
	multithreading_test()