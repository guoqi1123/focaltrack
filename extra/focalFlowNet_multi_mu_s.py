# This code does general focalflow given parameters using tensorflow
# Author: Qi Guo, Harvard University
# Email: qguo@seas.harvard.edu
# All Rights Reserved

import tensorflow as tf
import numpy as np
import cv2
from scipy import signal
from utils import *
import pdb

PADDING = 'VALID'
KEY_RANGE = {
	'raw' 			: [0,255],
	'gray' 			: [0,255],
	'test' 			: [0,255],
	'I_0' 			: [0,255],
	'I_1' 			: [0,255],
	'I_2' 			: [0,255],
}
class focalFlowNet_multi_mu_s(object):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'fx': np.array([[0.5,0,-0.5]]),
			'fy': np.array([[0.5],[0],[-0.5]]),
			'ft': np.array([[[-0.5,0,0.5]]]),
			# convolution window
			'separable' : True, # indicator of separability of conv
			'w' : np.ones((241,241)), # unseparable window
			'wx': np.ones((1,345)), # separable window
			'wy': np.ones((345,1)), # separable window
			# optical parameters
			'ratio' : 1e0, #for numerical stability
			'Sigma' : 0.001, #standard deviation of the isotropic filter, in mm
			'mu_s' : 130e-3, #sensor distance
			'f' : 100e-3, #focal distance
			'Z_0': 0, #Zero point of depth
			# other parameters
			'szx_sensor': 200,
			'szy_sensor': 200,
			'outs': ['Z', 'xdot', 'ydot', 'zdot'],
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

		# Change configurations
		for k in cfg.keys():
			self.cfg[k] = cfg[k]

		self.resolution = (self.cfg['szy_sensor'], self.cfg['szx_sensor'])		
		self.vars = {}
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

		self.image_to_show = ['I_c','Z','Z_gt']
		self.netName = "Focal Flow"

	"""imports a batch of frame into """
	def input_images(self, I, Z, mu_s):
		# import data into the network
		input_dict = {
			self.I_in: I,
			self.Z_in: Z,
			self.mu_s: mu_s,
		}
		self.session.run(self.input_data, input_dict)
		return

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""
	def build_graph(self):
		with self.graph.as_default():
			#unpack parameters for ease of typing
			mu_s = tf.Variable(self.cfg['mu_s'], dtype = tf.float32)
			self.mu_s = tf.Variable(self.cfg['mu_s'], dtype = tf.float32)

			# notice that we actually use sigma in pixel unit: sigma_pix
			sigma = self.cfg['Sigma']
			sigma_pix = sigma / self.cfg['pix_size']
			mu_f = 1 / (1 / self.cfg['f'] - 1 / mu_s)
			ratio = self.cfg['ratio']

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

			c_idx = (self.cfg['ft'].shape[2]-1)/2
			I_c = I[:,:,c_idx]
			
			_XX,_YY = np.meshgrid(
				np.arange(self.resolution[1]), 
				np.arange(self.resolution[0])
			) 
			#Center the coordinates
			_XX = (_XX - (self.resolution[1] - 1)/2)/ratio
			_YY = (_YY - (self.resolution[0] - 1)/2)/ratio

			XX = tf.constant(_XX, dtype=np.float32)
			YY = tf.constant(_YY, dtype=np.float32)
	
			# Generate the differential images
			I_t = dIdt(I, tf.constant(self.cfg['ft'],dtype=np.float32))
			I_x = dIdx(I_c, tf.constant(self.cfg['fx'],dtype=np.float32))
			I_y = dIdy(I_c, tf.constant(self.cfg['fy'],dtype=np.float32))
			I_xx = dIdx(I_c, tf.constant(self.cfg['fxx'],dtype=np.float32))
			I_yy = dIdy(I_c, tf.constant(self.cfg['fyy'],dtype=np.float32))
			I_xy = dIdx(I_c, tf.constant(self.cfg['fxy'],dtype=np.float32))
			I_yx = dIdx(I_c, tf.constant(self.cfg['fyx'],dtype=np.float32))
			xI_x = XX*I_x
			yI_y = YY*I_y

			# Convolution and windowing
			A = {}
			w_A = {} #windowed _A
		
			A['11'] = I_x*I_x;
			A['12'] = I_x*I_y;
			A['13'] = I_x*(xI_x + yI_y)
			A['14'] = I_x*(I_xx + I_yy)
	
			A['22'] = I_y*I_y
			A['23'] = I_y*(xI_x + yI_y)
			A['24'] = I_y*(I_xx + I_yy)
	
			A['33'] = (xI_x + yI_y)**2.0
			A['34'] = (xI_x + yI_y)*(I_xx + I_yy)
	
			A['44'] = (I_xx + I_yy)**2.0
		
			# I take the negative of b
			b  = [
				I_x*I_t, 
				I_y*I_t, 
				(xI_x + yI_y)*I_t, 
				(I_xx + I_yy)*I_t
			]

			#non-flat A's for visualizing during debugging (looks like an image)
			ATA = {} 
			if self.cfg['separable']:
				for k in A.keys():
					y = separable_window(\
						A[k], 
						tf.constant(self.cfg['wx'],dtype=tf.float32), 
						tf.constant(self.cfg['wy'],dtype=tf.float32),
						padding = PADDING,
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
						tf.constant(self.cfg['w'],dtype=tf.float32),
						padding = PADDING,
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
				for i in range(4):
					y = separable_window(
						b[i], 
						tf.constant(self.cfg['wx'],dtype=tf.float32), 
						tf.constant(self.cfg['wy'],dtype=tf.float32),
						padding = PADDING,
					)
					w_b.append( tf.reshape(y, [-1]) )
			else:
				for i in range(4):
					y = unseparable_window(
						b[i], 
						tf.constant(self.cfg['w'],dtype=tf.float32),
						padding = PADDING,
					)
					w_b.append( tf.reshape(y, [-1]) )
			
			ATA_flat_inv_dict, beta_flat, ATA_flat_det = inv4x4sym(w_A)
			index_4x4 = \
				['%d%d'%(i+1,j+1) for i in range(4) for j in range(4)]
			
			ATA_det = tf.reshape(ATA_flat_det, new_res)
			ATA_det_log = tf.log(ATA_det)
			
			ATA_flat_inv = tf.transpose(
				tf.pack( [[ATA_flat_inv_dict[k] for k in index_4x4]] ) 
			)

			ATb_flat = tf.transpose(tf.pack(w_b))
			ATA_inv = tf.reshape(ATA_flat_inv, [-1, 4, 4])
			ATb = tf.reshape(ATb_flat, [-1, 4, 1])
			
			U_flat = tf.batch_matmul(ATA_inv, ATb)
			U = tf.reshape(U_flat, new_res + [4])
			
			k = (mu_s*sigma_pix)**2.0/ratio
			Z = k*U[:,:,2]/(k*U[:,:,2]/mu_f +mu_f*U[:,:,3]);
			
			xdot = -U[:,:,0]*Z/mu_s
			ydot = -U[:,:,1]*Z/mu_s
			zdot = -U[:,:,2]*Z

			#save references to required I/O
			self.vars['I'] = I
			self.vars['I_c'] = I_c
			self.vars['Z'] = Z
			self.vars['Z_gt'] = Z_gt0
			self.vars['xdot'] = xdot
			self.vars['ydot'] = ydot
			self.vars['zdot'] = zdot

			#valid region
			self.valid_region()

			#save a reference for easy debugging purposes - there are some 
			#automatic ways of pulling this data out of the graph but this 
			#is much easier when prototyping
			self.vars['t'] = I_t
			self.vars['x'] = I_x
			self.vars['y'] = I_y
			self.vars['xx'] = I_xx
			self.vars['yy'] = I_yy
			self.vars['xy'] = I_xy
			self.vars['yx'] = I_yx
			self.vars['xI_x'] = xI_x
			self.vars['yI_y'] = yI_y		
			self.vars['ATA_det'] = ATA_det_log
			self.vars['ATb'] = []
			self.vars['U'] = []
			self.vars['ATA'] = {}
			self.vars['ATA^-1'] = {}
			#make it easy to see high dimensional per pixel variables
			for k in ["%d%d"%(i+1,j+1) for i in range(4) for j in range(4)]:
				self.vars['ATA'][k] = ATA[k]
				self.vars['ATA^-1'][k] = tf.reshape(ATA_flat_inv_dict[k], new_res)
			for i in range(4):
				self.vars['U'].append(U[:,:,i])
				self.vars['ATb'].append(tf.reshape(w_b[i], new_res))

			#add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in),
				mu_s.assign(self.mu_s),
			)
			#do not add anything to the compute graph after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	def regular_output(self, log = False):
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars[k]

		self.results = self.session.run(res_dict)	
		rng = {}
		for k in self.image_to_show:
			if k in KEY_RANGE.keys():
				rng[k] = KEY_RANGE[k]
			else:
				rng[k] = [np.NaN, np.NaN]
		self.cache['draw'] = tile_image(\
									I = self.results, \
									rng = rng, \
									log = log, \
									title = "Regular Output", \
								)
		cv2.imshow(self.netName, self.cache['draw'])

	def query_results(self, query_list):
		res_dict = {}
		for k in query_list:
			res_dict[k] = self.vars[k]
		self.results = self.session.run(res_dict)
		return self.results

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
			
	"""destructor: free up resources when done"""
	def __del__(self):
		cv2.destroyAllWindows()

