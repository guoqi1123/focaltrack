#!/usr/bin/python2.7

import argparse

import cv2
import flycapture2 as fc2
import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import json
import os, glob

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
I_cache = 0
I_in_input = 0
I_lap_batch_input = 0
input_dict = 0
ending_key = 'c'

#the range for different outputs, range set to NaN means auto-ranging
DEPTH_RANGE = [-0.88,-0.48]
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

		self.t = []
		self.draw = None
		self.cache = {}
		self.vars = {}
		self.frames = 0
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
		
		self.I_cache[:,:,self.idx] = self.cache['gray']
		self.idx = 1 - self.idx

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

		self.sample_images = None
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)

		self.data_type = tf.float32

		self.I_cache = np.zeros(self.resolution[0]+(2,), dtype = np.uint8)
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
			self.regular_output_layers()
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
				scipy.misc.imsave('./models/demoshot/edge.png', \
					(self.cache['draw']*255).astype(np.uint8)
				)
				
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

			I_batch = []
			I_lap_batch = []

			Z_0 = []
			a0 = []
			a1 = []
			gauss = []
			ft = []
			fave = []

			I_t = []
			I_lap = []

			u_1 = []
			u_2 = []
			u_3 = []
			u_4 = []
			Z = []

			# depth computation
			# I used to be (960, 600, 2or 3) we change it to (2or3, 960, 600) 
			tmp_I = tf.transpose(I, perm=[2,0,1])
			for i in range(len(self.cfg)):
				# initialize variables				
				"""Input parameters"""
				Z_0.append(tf.constant(self.cfg[i]['Z_0'], dtype = tf.float32))
				a0.append(tf.constant(self.cfg[i]['a0'], dtype = tf.float32))
				a1.append(tf.constant(self.cfg[i]['a1'], dtype = tf.float32))

				gauss.append(tf.Variable(self.cfg[i]['gauss'], dtype = tf.float32))
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
				u_3.append(a0[i] * a1[0] * u_1[i])
				u_4.append(-u_2[i] + a0[i] * u_1[i] + 1e-5)
				Z.append(u_3[i]/(u_4[i] + 1e-5) + Z_0[0])
				
				#save references to required I/O]
				self.vars.append({})
				self.vars[i]['I'] = I_batch[i]
				
				# unwindowed version
				self.vars[i]['u_1'] = u_1[i]
				self.vars[i]['u_2'] = u_2[i]
				self.vars[i]['u_3'] = u_3[i]
				self.vars[i]['u_4'] = u_4[i]
				self.vars[i]['Z'] = Z[i]

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[i]['conf_func']+'()')
			# self.w3_baseline_conf()

			# align depth and confidence maps
			self.align_maps_linear(['Z','conf'])
			# self.vars_align['conf'] = 1/self.vars_align['unc']			
			# self.soft_thresholding()

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
				I.assign(self.I_in),
			)

			#do not add anything to the compute graph 
			#after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	def w3r_baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability,
		# use the working range to cut the confidence
		# and use weight for each layer
		# the windowed flag indicates whether we compute windowed
		w_bc = [] # the weight of baseline confidence for each layer
		w_bc1 = []
		w_bc2 = []
		lo = [] # the low threshold of the wr
		hi = [] # the high threshold of the wr
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

			# working ranges
			lo.append(
				tf.Variable(
					self.cfg[i]['lo'],
					dtype = tf.float32
				)
			)
			hi.append(
				tf.Variable(
					self.cfg[i]['hi'],
					dtype = tf.float32
				)
			)

			# unwindowed version
			conf_tmp = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars[i]['u_3']**2 + \
					w_bc1[i] * self.vars[i]['u_4']**2 + \
					w_bc2[i] * self.vars[i]['u_3']*self.vars[i]['u_4'] + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Z = self.vars[i]['Z']
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))

			# final confidence
			self.vars[i]['conf'] = conf_tmp * cut
			self.vars[i]['unc'] = 1/(self.vars[i]['conf']+1e-20)
			self.vars[i]['cut'] = cut
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['w_bc1'] = w_bc1[i]
			self.vars[i]['w_bc2'] = w_bc2[i]
			self.vars[i]['lo'] = lo[i]
			self.vars[i]['hi'] = hi[i]

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
			# unwindowed version
			conf_tmp = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars[i]['u_3']**2 + \
					w_bc1[i] * self.vars[i]['u_4']**2 + \
					w_bc2[i] * self.vars[i]['u_3']*self.vars[i]['u_4'] + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				)

			# final confidence
			self.vars[i]['conf'] = conf_tmp
			self.vars[i]['unc'] = 1/(self.vars[i]['conf']+1e-5)
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['w_bc1'] = w_bc1[i]
			self.vars[i]['w_bc2'] = w_bc2[i]

		return

	def soft_thresholding(self):
		conf = []
		k = 50
		lo = [] # the low threshold of the wr
		hi = [] # the high threshold of the wr
		for i in range(len(self.cfg)):
			# working ranges
			lo.append(
				tf.Variable(
					self.cfg[i]['lo'],
					dtype = tf.float32
				)
			)
			hi.append(
				tf.Variable(
					self.cfg[i]['hi'],
					dtype = tf.float32
				)
			)

			Z = self.vars_align['Z'][:,:,i]
			conf.append(self.vars_align['conf'][:,:,i])
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))
			conf[i] = conf[i] * cut

		self.vars_align['conf'] = tf.pack(conf,2)
		return




	def valid_windowed_region_fuse(self):
		# cut out the bad part
		vars_to_cut = [\
			'Z','conf'\
		]
		rows_cut = 1
		cols_cut = 1
		for i in range(len(self.cfg)):
			rows_cut *= int((self.cfg[i]['gauss'].shape[0]-1)/2)
			cols_cut *= int((self.cfg[i]['gauss'].shape[0]-1)/2)

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

		myThread.acquire()
		self.input_dict[self.I_in] = I_cache
		myThread.release()

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

	def regular_output(self):
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
			0.99
		)
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

		conf_thre = 0.99
		self.results['Z'] = self.pseudo_color_Z(\
			self.results['Z'],\
			self.results['conf'],\
			DEPTH_RANGE[0],\
			DEPTH_RANGE[1],\
			conf_thre
		)
		# print out the Z in each layer
		for i in range(len(self.cfg)):
			self.results['Z'+str(i)] = self.results['Z_align'][:,:,i]
			self.results['conf'+str(i)] = self.results['conf_align'][:,:,i]
			self.results['Z'+str(i)] = self.pseudo_color_Z(\
				self.results['Z'+str(i)],\
				self.results['conf'+str(i)],\
				DEPTH_RANGE[0],\
				DEPTH_RANGE[1],\
				conf_thre
			)
			self.image_to_show.append('Z'+str(i))
			self.image_to_show.append('conf'+str(i))

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
		rows_cut = 1
		cols_cut = 1
		for i in range(len(self.cfg)):
			rows_cut *= int((self.cfg[i]['gauss'].shape[0]-1)/2)
			cols_cut *= int((self.cfg[i]['gauss'].shape[0]-1)/2)

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
	a = Camera()

	a.start()

	time.sleep(1)

	# initialize the pulsecam processor
	cfg_file = "./opt_results/pyConfLensFlowNetFast/"+\
		"1x1t-text34-setup5-py4-w3r-whole.pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']
	b = PulseCamProcessor(cfg)

	b.start()
	
	a.join()

	b.join()

if __name__ == "__main__":
	# debug_test()
	multithreading_test()