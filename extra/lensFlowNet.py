# This code does general focalflow given parameters using tensorflow
# Author: Qi Guo, Harvard University
# Email: qguo@seas.harvard.edu
# All Rights Reserved

import tensorflow as tf
import numpy as np
import cv2
from scipy import signal
import pdb
import pickle
from utils import *
from focalFlowNet import focalFlowNet
from focalFlowNet import KEY_RANGE

PADDING = 'SAME'
class lensFlowNet(focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'lap': np.array([[0,1,0],[1,-4,1],[0,1,0]]),
			'ft': np.array([[[-0.5,0.5]]]),
			'fave': np.array([[[0.5,0.5]]]),
			'a0' : 130e-3, #sensor distance
			'a1' : 100e-3, #focal distance
			'Z_0': 0,
			# convolution window
			'separable' : True, # indicator of separability of conv
			'w' : np.ones((35,35)), # unseparable window
			'wx': np.ones((1,345)), # separable window
			'wy': np.ones((345,1)), # separable window
			# other parameters
			'szx_sensor': 200,
			'szy_sensor': 200,
			'outs': ['Z'],
		}

		# Change configurations according to the input
		for k in cfg.keys():
			self.cfg[k] = cfg[k]

		self.cfg['P'] = 1/self.cfg['f']

		self.resolution = (self.cfg['szy_sensor'], self.cfg['szx_sensor'])		
		self.vars = {}
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

		self.image_to_show = ['Z','Z_gt','Z_err','conf']
		self.netName = "Lens Flow"

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""
	def build_graph(self):
		with self.graph.as_default():
			I_init = np.zeros(
				self.resolution+(self.cfg['ft'].shape[2],), 
				dtype = np.float32
			)
			Z_init = np.zeros(
				self.resolution,
				dtype = np.float32
			)
			self.I_in = tf.Variable(I_init)
			self.Z_in = tf.Variable(Z_init)
			I = tf.Variable(I_init)

			Z_gt = tf.Variable(Z_init)
			Z_0 = tf.Variable(self.cfg['Z_0'])
			Z_gt0 = Z_gt - Z_0

			I_trans = tf.transpose(I, perm=[2,0,1])		

			if self.cfg['separable']:
				wx = tf.constant(
					self.cfg['wx'], dtype = tf.float32
				)
				wy = tf.constant(
					self.cfg['wy'], dtype = tf.float32
				)
			else:
				w = tf.constant(
					self.cfg['w'], dtype = tf.float32
				)	
		
			a0 = tf.constant(self.cfg['a0'], dtype = tf.float32)
			a1 = tf.constant(self.cfg['a1'], dtype = tf.float32)

			# Generate the differential images
			I_t = dIdt(I,tf.constant(self.cfg['ft'],dtype=tf.float32))

			# we compute the laplacian of the image in a batch
			I_lap_batch = dIdx_batch(
				I_trans,tf.constant(self.cfg['lap'],dtype=tf.float32)
			)
			# conduct averaging
			I_lap_batch_trans = tf.transpose(I_lap_batch, perm = [1,2,0])
			I_lap = dIdt(
				I_lap_batch_trans, 
				tf.constant(self.cfg['fave'],dtype=tf.float32)
			)

			if self.cfg['separable']:
				u_1 = separable_window(
					I_lap*I_lap, 
					wx, 
					wy, 
					PADDING,
				)
				u_2 = separable_window(
					I_t*I_lap, 
					wx, 
					wy, 
					PADDING,
				)
			else:
				u_1 = unseparable_window(
					I_lap*I_lap, 
					w, 
					PADDING,
				)
				u_2 = unseparable_window(
					I_t*I_lap, 
					w,
					PADDING,
				)
			u_1w = tf.squeeze(u_1, [0,3])
			u_2w = tf.squeeze(u_2, [0,3])
			u_3w = a0 * a1 * u_1w
			u_4w = -u_2w + a0 * u_1w

			u_1 = I_lap
			u_2 = I_t
			u_3 = a0 * a1 * u_1
			u_4 = -u_2 + a0 * u_1
			
			Z = u_3w / u_4w
			uncw = tf.sqrt(u_3w**2 + u_4w**2 + u_4w**4+1e-10)/(u_4w**2+1e-20)
			unc = tf.sqrt(u_3**2 + u_4**2 + u_4**4+1e-10)/(u_4**2+1e-20)
			confw = 1/uncw
			conf = 1/unc

			
			#save references to required I/O
			self.vars['I1'] = I[:,:,0]
			self.vars['I2'] = I[:,:,1]
			self.vars['I'] = I
			self.vars['Z'] = Z
			self.vars['unc'] = unc
			self.vars['uncw'] = uncw
			self.vars['conf'] = conf
			self.vars['confw'] = confw
			self.vars['Z_gt'] = Z_gt0
			self.vars['Z_err'] = tf.abs(Z - Z_gt0)
			self.vars['u_1'] = u_1
			self.vars['u_2'] = u_2
			self.vars['u_3'] = u_3
			self.vars['u_4'] = u_4
			self.vars['u_1w'] = u_1w
			self.vars['u_2w'] = u_2w
			self.vars['u_3w'] = u_3w
			self.vars['u_4w'] = u_4w
			self.vars['a0'] = a0
			self.vars['a1'] = a1

			#valid region
			self.valid_region()

			#save a reference for easy debugging purposes - there are some 
			#automatic ways of pulling this data out of the graph but this 
			#is much easier when prototyping
			self.vars['t'] = I_t
			self.vars['lap'] = I_lap

			#add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in),
			)
			#do not add anything to the compute graph after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	def valid_region(self):
		# find out valid regions for Z, Z_gt
		vars_to_cut = ['u_1w','u_2w','u_3w','u_4w','Z','uncw','confw']
		if PADDING == 'SAME':
			if self.cfg['separable']:
				rows_cut = int(
					(self.cfg['wx'].shape[1]-1)/2 + 
					(self.cfg['lap'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['wy'].shape[0]-1)/2 + 
					(self.cfg['lap'].shape[0]-1)/2
				)
			else:
				rows_cut = int(
					(self.cfg['w'].shape[1]-1)/2 + 
					(self.cfg['lap'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['w'].shape[0]-1)/2 + 
					(self.cfg['lap'].shape[0]-1)/2
				)
			cols = self.cfg['szy_sensor']
			rows = self.cfg['szx_sensor']
		elif PADDING == 'VALID':
			rows_cut = int(
				(self.cfg['lap'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['lap'].shape[0]-1)/2
			)
			cols = self.cfg['szy_sensor'] - (self.cfg['wy'].shape[0]-1)
			rows = self.cfg['szx_sensor'] - (self.cfg['wx'].shape[1]-1)
		
		for var in vars_to_cut:
			self.vars[var+'_valid'] = self.vars[var][
				cols_cut:cols-cols_cut,
				rows_cut:rows-rows_cut
			]
			self.vars[var+'_valid_flat'] = tf.reshape(
				self.vars[var+'_valid'], [-1]
			)

		# cut 'Z_gt'
		if self.cfg['separable']:
			rows_cut = int(
				(self.cfg['wx'].shape[1]-1)/2 + 
				(self.cfg['lap'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['wy'].shape[0]-1)/2 + 
				(self.cfg['lap'].shape[0]-1)/2
			)
		else:
			rows_cut = int(
				(self.cfg['w'].shape[1]-1)/2 + 
				(self.cfg['lap'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['w'].shape[0]-1)/2 + 
				(self.cfg['lap'].shape[0]-1)/2
			)
		cols = self.cfg['szy_sensor']
		rows = self.cfg['szx_sensor']
		self.vars['Z_gt_valid'] = self.vars['Z_gt'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['Z_gt_valid_flat'] = tf.reshape(
			self.vars['Z_gt_valid'], [-1]
		)

		# cut u_1, u_2,
		vars_to_cut = ['u_1','u_2','u_3','u_4','unc','conf']
		rows_cut = int(
			(self.cfg['lap'].shape[1]-1)/2
		)
		cols_cut = int(
			(self.cfg['lap'].shape[0]-1)/2
		)
		cols = self.cfg['szy_sensor']
		rows = self.cfg['szx_sensor']
		for var in vars_to_cut:
			self.vars[var+'_valid'] = self.vars[var][
				cols_cut:cols-cols_cut,
				rows_cut:rows-rows_cut
			]
			self.vars[var+'_valid_flat'] = tf.reshape(
				self.vars[var+'_valid'], [-1]
			)
		return 