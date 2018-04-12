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

PADDING = 'VALID'
class lensFlowNet(focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'fx': np.array([[0.5,0,-0.5]]),
			'fy': np.array([[0.5],[0],[-0.5]]),
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

		self.cfg['fxx'] = signal.convolve2d(
			self.cfg['fx'],self.cfg['fx'],mode='full'
		)
		self.cfg['fyy'] = signal.convolve2d(
			self.cfg['fy'],self.cfg['fy'],mode='full'
		)
		self.cfg['fxy'] = signal.convolve2d(
			self.cfg['fx'],self.cfg['fy'],mode='full'
		)
		self.cfg['fyx'] = signal.convolve2d(
			self.cfg['fy'],self.cfg['fx'],mode='full'
		)
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

		self.image_to_show = ['Z','Z_gt']
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
				wx = tf.Variable(
					self.cfg['wx'], dtype = tf.float32
				)
				wy = tf.Variable(
					self.cfg['wy'], dtype = tf.float32
				)
			else:
				w = tf.Variable(
					self.cfg['w'], dtype = tf.float32
				)	
		
			a0 = tf.constant(self.cfg['a0'], dtype = tf.float32)
			a1 = tf.constant(self.cfg['a1'], dtype = tf.float32)

			# Generate the differential images
			I_t = dIdt(I,tf.constant(self.cfg['ft'],dtype=tf.float32))

			# we compute the laplacian of the image in a batch
			I_xx = dIdx_batch(
				I_trans,tf.constant(self.cfg['fxx'],dtype=tf.float32)
			)
			I_yy = dIdy_batch(
				I_trans,tf.constant(self.cfg['fyy'],dtype=tf.float32)
			)
			I_xy = dIdx_batch(
				I_trans,tf.constant(self.cfg['fxy'],dtype=tf.float32)
			)
			I_yx = dIdx_batch(
				I_trans,tf.constant(self.cfg['fyx'],dtype=tf.float32)
			)
			I_lap_batch = I_xx + I_yy
			# conduct averaging
			I_lap_batch_trans = tf.transpose(I_lap_batch, perm = [1,2,0])
			I_lap = dIdt(
				I_lap_batch_trans, 
				tf.constant(self.cfg['fave'],dtype=tf.float32)
			)

			# Convolution and windowing
			A = {}
			w_A = {} #windowed _A
		
			A['11'] = I_lap*I_lap;

			# compute ATA and ATb
			# I take the negative of b
			b  = [
				I_lap*I_t, 
			]

			#non-flat A's for visualizing during debugging (looks like an image)
			ATA = {} 
			if self.cfg['separable']:
				for k in A.keys():
					y = separable_window(\
						A[k], 
						wx, 
						wy,
						PADDING,
					)
					ATA[k] = tf.squeeze(y)
					w_A[k] = tf.reshape(y, [-1])
					#exploit symmetry
					if not (k[::-1] in ATA):
						ATA[k[::-1]] = ATA[k]
						w_A[k[::-1]] = w_A[k]
			else:
				for k in A.keys():
					y = unseparable_window(\
						A[k], 
						w,
						PADDING,
					)
					ATA[k] = tf.squeeze(y)
					w_A[k] = tf.reshape(y, [-1])
					#exploit symmetry
					if not (k[::-1] in ATA):
						ATA[k[::-1]] = ATA[k]
						w_A[k[::-1]] = w_A[k]
			
			new_res = ATA['11'].get_shape().as_list()  #save post convolution image resolution

			w_b = []
			if self.cfg['separable']:
				for i in range(1):
					y = separable_window(
						b[i], 
						wx, 
						wy, 
						PADDING,
					)
					w_b.append( tf.reshape(y, [-1]) )
			else:
				for i in range(1):
					y = unseparable_window(
						b[i], 
						w, 
						PADDING,
					)
					w_b.append( tf.reshape(y, [-1]) )
			
			ATA_flat_inv_dict = {
				'11': 1/w_A['11']
			}
			index_1x1 = \
				['%d%d'%(i+1,j+1) for i in range(1) for j in range(1)]
			
			ATA_flat_inv = tf.transpose(
				tf.pack( [[ATA_flat_inv_dict[k] for k in index_1x1]] ) 
			)

			ATb_flat = tf.transpose(tf.pack(w_b))
			ATA_inv = tf.reshape(ATA_flat_inv, [-1, 1, 1])
			ATb = tf.reshape(ATb_flat, [-1, 1, 1])
			
			U_flat = tf.batch_matmul(ATA_inv, ATb)
			U = tf.reshape(U_flat, new_res + [1])

			Z = a0*a1 / (-U[:,:,0]+a0)
			
			#save references to required I/O
			self.vars['I1'] = I[:,:,0]
			self.vars['I2'] = I[:,:,1]
			self.vars['I'] = I
			self.vars['Z'] = Z
			self.vars['Z_gt'] = Z_gt0

			#valid region
			self.valid_region()

			#save a reference for easy debugging purposes - there are some 
			#automatic ways of pulling this data out of the graph but this 
			#is much easier when prototyping
			self.vars['U'] = U[:,:,0]
			self.vars['t'] = I_t
			self.vars['xx'] = I_xx
			self.vars['yy'] = I_yy
			self.vars['xy'] = I_xy
			self.vars['yx'] = I_yx
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
		if PADDING == 'SAME':
			if self.cfg['separable']:
				rows_cut = int(
					(self.cfg['wx'].shape[1]-1)/2 + 
					(self.cfg['fxx'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['wy'].shape[0]-1)/2 + 
					(self.cfg['fyy'].shape[0]-1)/2
				)
			else:
				rows_cut = int(
					(self.cfg['w'].shape[1]-1)/2 + 
					(self.cfg['fxx'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['w'].shape[0]-1)/2 + 
					(self.cfg['fyy'].shape[0]-1)/2
				)
			cols = self.cfg['szy_sensor']
			rows = self.cfg['szx_sensor']
		elif PADDING == 'VALID':
			rows_cut = int(
				(self.cfg['fxx'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['fyy'].shape[0]-1)/2
			)
			cols = self.cfg['szy_sensor'] - (self.cfg['wy'].shape[0]-1)
			rows = self.cfg['szx_sensor'] - (self.cfg['wx'].shape[1]-1)
		
		self.vars['Z_valid'] = self.vars['Z'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['Z_valid_flat'] = tf.reshape(
			self.vars['Z_valid'], [-1]
		)

		# cut 'Z_gt'
		if self.cfg['separable']:
			rows_cut = int(
				(self.cfg['wx'].shape[1]-1)/2 + 
				(self.cfg['fxx'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['wy'].shape[0]-1)/2 + 
				(self.cfg['fyy'].shape[0]-1)/2
			)
		else:
			rows_cut = int(
				(self.cfg['w'].shape[1]-1)/2 + 
				(self.cfg['fxx'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['w'].shape[0]-1)/2 + 
				(self.cfg['fyy'].shape[0]-1)/2
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
		return 