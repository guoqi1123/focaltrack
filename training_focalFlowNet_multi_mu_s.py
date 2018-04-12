# This code trains the focal flow network
# using the data simulated in simulation.py
# Author: Qi Guo, Harvard University
# Email: qguo@seas.harvard.edu
# All Rights Reserved

# The training data should be in 
# ./simulation_data/training_focalFlowNet

import tensorflow as tf
import numpy as np
import cv2
from scipy import signal
import pdb
import pickle
import matplotlib.pyplot as plt
from focalFlowNet_multi_mu_s import focalFlowNet_multi_mu_s
from focalFlowNet_multi_mu_s import KEY_RANGE
from utils import *

# Global padding option
PADDING = 'VALID'
class training_focalFlowNet_multi_mu_s(focalFlowNet_multi_mu_s):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default initialization of parameters
		self.cfg = {
			# finite difference coefficient
			'fx': np.array([[0.5,0,-0.5]]),
			'fy': np.array([[0.5],[0],[-0.5]]),
			'ft': np.array([[[-0.5,0,0.5]]]),
			# convolution window
			'separable' : True, # indicator of separability of conv
			'w' : np.ones((35,35)), # unseparable window
			'wx': np.ones((1,345)), # separable window
			'wy': np.ones((345,1)), # separable window
			# optical parameters
			'ratio' : 1e0, #for numerical stability
			'Sigma' : 0.001, #standard deviation of the isotropic filter, in m
			'dSigma_ratio': 0.000001, #adjust the scale to converge faster
			'mu_s' : 130e-3, #sensor distance
			'dmu_s_ratio': 0.000001, #adjust the scale to converge faster
			'f' : 100e-3, #focal distance
			'df_ratio': 0.000001, #adjust the scale to converge faster
			'Z_0': 0, #the zero point of depth
			'dZ_0_ratio': 0.000001, #adjust the scale to converge faster
			# other parameters
			'szx_sensor': 200,
			'szy_sensor': 200,
			'outs': ['Z', 'xdot', 'ydot', 'zdot'],
			'learn_rate': 0.01,
			'err_func': ['huber_err',]
		}

		self.cfg['fxx'] = signal.convolve2d(
			self.cfg['fx'],self.cfg['fx'],mode='full'
		)
		self.cfg['fyy'] = signal.convolve2d(
			self.cfg['fy'],self.cfg['fy'],mode='full'
		)

		# Change configurations
		for k in cfg.keys():
			self.cfg[k] = cfg[k]

		self.resolution = (self.cfg['szy_sensor'], self.cfg['szx_sensor'])		
		self.vars = {}
		self.der = {}
		self.der_f = {}
		self.input_der = {}
		self.input_var = {}
		self.ave_der = {}
		self.list_der = []
		self.loss = 0
		self.cur_err_idx = 0
		self.cur_err_func = self.cfg['err_func'][0]

		if self.cfg['separable']:
			self.der_dict = {
				'fx':'dLdfx',
				'fy':'dLdfy',
				'fxx':'dLdfxx',
				'fyy':'dLdfyy',
				'wx':'dLdwx',
				'wy':'dLdwy',
				'Sigma':'dLdSigma',
				'mu_s':'dLdmu_s',
				'f':'dLdf',
				'Z_0':'dZ_0df',
			}
			self.var_dict = {
				'dLdfx':'fx',
				'dLdfy':'fy',
				'dLdfxx':'fxx',
				'dLdfyy':'fyy',
				'dLdwx':'wx',
				'dLdwy':'wy',
				'dLdSigma':'Sigma',
				'dLdmu_s':'mu_s',
				'dLdf':'f',
				'dZ_0df':'Z_0',
			}
		else:
			self.der_dict = {
				'fx':'dLdfx',
				'fy':'dLdfy',
				'fxx':'dLdfxx',
				'fyy':'dLdfyy',
				'w':'dLdw',
				'Sigma':'dLdSigma',
				'mu_s':'dLdmu_s',
				'f':'dLdf',
				'Z_0':'dZ_0df',
			}
			self.var_dict = {
				'dLdfx':'fx',
				'dLdfy':'fy',
				'dLdfxx':'fxx',
				'dLdfyy':'fyy',
				'dLdw':'w',
				'dLdSigma':'Sigma',
				'dLdmu_s':'mu_s',
				'dLdf':'f',
				'dZ_0df':'Z_0',
			}
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

		self.image_to_show = ['I_c']
		self.netName = "Training Focal Flow"

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""
	def build_graph(self):
		with self.graph.as_default():
			"""Input images and depths"""
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
			Z_0 = tf.Variable(self.cfg['Z_0'], dtype = tf.float32)
			Z_gt = tf.Variable(Z_init)
			Z_gt0 = Z_gt + self.cfg['Z_0'] - Z_0

			"""Input parameters"""
			#unpack parameters and turn them into tf.Variable
			mu_s = tf.Variable(self.cfg['mu_s'], dtype = tf.float32)
			self.mu_s = tf.Variable(self.cfg['mu_s'], dtype = tf.float32)
			mu_s0 = mu_s + self.cfg['Z_0'] - Z_0

			f = tf.Variable(self.cfg['f'], dtype = tf.float32)
			# notice that we actually use Sigma in pixel unit: Sigma_pix
			Sigma = tf.Variable(self.cfg['Sigma'], dtype = tf.float32)
			Sigma_pix = Sigma / self.cfg['pix_size']
			ratio = tf.Variable(self.cfg['ratio'], dtype = tf.float32)
			ft = tf.Variable(self.cfg['ft'], dtype = tf.float32)
			fx = tf.Variable(self.cfg['fx'], dtype = tf.float32)
			fy = tf.Variable(self.cfg['fy'], dtype = tf.float32)
			# if fxx is in the list of vars to be optimized, make it a variable
			# if not, make it a tensor depended on fx
			if 'dLdfxx' in self.cfg['der_var'][self.cur_err_func]:
				fxx = tf.Variable(self.cfg['fxx'], dtype = tf.float32)
			else:
				hf_row = np.int((self.cfg['fx'].shape[0]-1)/2)
				hf_col = np.int((self.cfg['fx'].shape[1]-1)/2)
				paddings = [[hf_row, hf_row],[hf_col, hf_col]]
				fx_pad = tf.pad(fx, paddings, mode='CONSTANT')
				fxx = dIdx(fx_pad, -fx)

			if 'dLdfxx' in self.cfg['der_var'][self.cur_err_func]:
				fyy = tf.Variable(self.cfg['fyy'], dtype = tf.float32)
			else:
				hf_row = np.int((self.cfg['fy'].shape[0]-1)/2)
				hf_col = np.int((self.cfg['fy'].shape[1]-1)/2)
				paddings = [[hf_row, hf_row],[hf_col, hf_col]]
				fy_pad = tf.pad(fy, paddings, mode='CONSTANT')
				fyy = dIdy(fy_pad, -fy)
			
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
			
			"""Initialize other variables based on inputs"""
			mu_f = 1 / (1 / f - 1 / mu_s0)
			self.cfg['mu_f'] = mu_f
			c_idx = int((self.cfg['ft'].shape[2]-1)/2)
			I_c = I[:,:,c_idx]
			XX,YY = np.meshgrid(
				np.arange(self.resolution[1]), np.arange(self.resolution[0])
			) 
			#Center the coordinates
			XX = (XX - (self.resolution[1] - 1)/2)/ratio
			YY = (YY - (self.resolution[0] - 1)/2)/ratio

			"""Input the final derivative"""
			der_pair = {}
			for key in self.der_dict.keys():
				self.input_der[self.der_dict[key]] = \
					tf.Variable(self.cfg[key], dtype = tf.float32)
				self.der_f[self.der_dict[key]] = \
					tf.Variable(self.cfg[key], dtype = tf.float32)
				der_pair[self.der_dict[key]] = (
					self.der_f[self.der_dict[key]] + 0.#turn it into a tensor
					, eval(key) 
				)
				self.input_var[self.der_dict[key]] = \
					tf.Variable(self.cfg[key], dtype = tf.float32)

	
			"""Computation starts here"""
			# Generate the differential images
			I_t = dIdt(I, ft)
			I_x = dIdx(I_c, fx)
			I_y = dIdy(I_c, fy)
			I_xx = dIdx(I_c, fxx)
			I_yy = dIdy(I_c, fyy)
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
				for i in range(4):
					y = separable_window(
						b[i], 
						wx, 
						wy, 
						PADDING,
					)
					w_b.append( tf.reshape(y, [-1]) )
			else:
				for i in range(4):
					y = unseparable_window(
						b[i], 
						w, 
						PADDING,
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
			
			k = (mu_s0*Sigma_pix)**2.0/ratio
			Z = k*U[:,:,2]/(k*U[:,:,2]/mu_f +mu_f*U[:,:,3]);
			
			xdot = -U[:,:,0]*Z/mu_s0
			ydot = -U[:,:,1]*Z/mu_s0
			zdot = -U[:,:,2]*Z
			
			#save references to required I/O
			self.vars['I'] = I
			self.vars['I_c'] = I_c
			self.vars['Z'] = Z
			self.vars['Z_gt'] = Z_gt0
			self.vars['xdot'] = xdot
			self.vars['ydot'] = ydot
			self.vars['zdot'] = zdot
			self.vars['Sigma_pix'] = Sigma_pix
			self.vars['U'] = U

			# compute error
			self.vars['loss_func'] = {}
			for ef in self.cfg['err_func']:
				self.vars['loss_func'][ef] = \
					eval('self.'+ef+'()')

			####################################################################
			# below are derivatives and optimization
			####################################################################
			#### derivatives w.r.t. filters
			self.der['dLdfx'] = {}
			self.der['dLdfy'] = {}
			self.der['dLdfxx'] = {}
			self.der['dLdfyy'] = {}
			self.der['dLdft'] = {}
			self.vars['fx'] = fx
			self.vars['fy'] = fy
			self.vars['fxx'] = fxx
			self.vars['fyy'] = fyy
			self.vars['ft'] = ft

			if self.cfg['separable']:
				self.der['dLdwx'] = {}
				self.der['dLdwy'] = {}
				self.vars['wx'] = wx
				self.vars['wy'] = wy
			else:
				self.der['dLdw'] = {}
				self.vars['w'] = w

			self.der['dLdSigma'] = {}
			self.der['dLdmu_s'] = {}
			self.der['dLdf'] = {}
			self.der['dLdZ_0'] = {}
			self.vars['Sigma'] = Sigma
			self.vars['mu_s'] = mu_s0
			self.vars['mu_f'] = mu_f
			self.vars['f'] = f
			self.vars['Z_0'] = Z_0

			for ef in self.cfg['err_func']:
				dLdfx, dLdfy, dLdfxx, dLdfyy, dLdft = \
					tf.gradients(self.vars['loss_func'][ef], [fx, fy, fxx, fyy, ft])
				num_inst = tf.to_float(tf.shape(self.vars['loss_func'][ef])[0])
				# restrict the symmetry:
				# dLdfx = dLdfy', dLdfx + reverse(dLdfx) = 0
				# this is validated by flipping the image will flip and negate the 
				# derivative, moreover it makes the derivative independent of 
				# brightness level
				dLdfx = dLdfx + tf.transpose(dLdfy)
				dLdfx = dLdfx - tf.reverse(dLdfx, dims = [True, True])
				dLdfx = dLdfx / 4. / num_inst
				dLdfy = tf.transpose(dLdfx)
				self.der['dLdfx'][ef] = dLdfx
				self.der['dLdfy'][ef] = dLdfy
				

				# restrict the symmetry:
				# dLdfxx = dLdfyy', dLdfxx - reverse(dLdfxx) = 0
				# ALSO sum(dLdfxx) = 0
				# this is validated by flipping the image will flip the derivative
				# moreover it makes the derivative independent of brightness level
				# it seems that the original derivative has a constant added to it
				# which deals with the brightness level
				dLdfxx = dLdfxx + tf.transpose(dLdfyy)
				dLdfxx = dLdfxx + tf.reverse(dLdfxx, dims = [True, True])
				dLdfxx = dLdfxx / 4. / num_inst
				dLdfxx = dLdfxx - tf.reduce_mean(dLdfxx)
				dLdfyy = tf.transpose(dLdfxx)
				self.der['dLdfxx'][ef] = dLdfxx
				self.der['dLdfyy'][ef] = dLdfyy
				

				# restrict the symmetry:
				# dLdft + tf.reverse(dLdft) = 0
				dLdft = dLdft - tf.reverse(dLdft, dims = [False, False, True])
				dLdft = dLdft / 2. / num_inst
				self.der['dLdft'][ef] = dLdft
				

				# #### derivatives w.r.t. windows
				# if self.cfg['separable']:
				# 	dLdwx, dLdwy = tf.gradients(
				# 		self.vars['loss_func'][ef], [wx, wy]
				# 	)
				# 	# restrict the symmetry
				# 	# dLdwx = dLdwy', dLdwx - reverse(dLdwx) = 0
				# 	dLdwx = dLdwx + tf.transpose(dLdwy)
				# 	dLdwx = dLdwx + tf.reverse(dLdwx, dims = [True, True])
				# 	dLdwx = dLdwx / 4. / num_inst
				# 	dLdwy = tf.transpose(dLdwx)
				# 	self.der['dLdwx'][ef] = dLdwx
				# 	self.der['dLdwy'][ef] = dLdwy
				# else:
				# 	dLdw, = tf.gradients(
				# 		self.vars['loss_func'][ef], [w, ]
				# 	)
				# 	# restrict the symmetry
				# 	# dLdw = dLdw', dLdw - reverse(dLdw) = 0
				# 	dLdw = dLdw + tf.transpose(dLdw)
				# 	dLdw = dLdw + tf.reverse(dLdw, dims = [True, True])
				# 	dLdw = dLdw / 4. / num_inst
				# 	self.der['dLdw'][ef] = dLdw
					

				#### derivative w.r.t. optical parameters
				dLdSigma, dLdmu_s, dLdf, dLdZ_0 = tf.gradients(
					self.vars['loss_func'][ef], [Sigma, mu_s0, f, Z_0]
				)
				dLdSigma = dLdSigma / num_inst * self.cfg['dSigma_ratio']
				dLdmu_s = dLdmu_s / num_inst * self.cfg['dmu_s_ratio']
				dLdf = dLdf / num_inst * self.cfg['df_ratio']
				dLdZ_0 = dLdZ_0 / num_inst * self.cfg['dZ_0_ratio']
				self.der['dLdSigma'][ef] = dLdSigma
				self.der['dLdmu_s'][ef] = dLdmu_s
				self.der['dLdf'][ef] = dLdf
				self.der['dLdZ_0'][ef] = dLdZ_0
			

			self.output_grads = {}
			for ef in self.cfg['err_func']:
				self.output_grads[ef] = {}
				for key in self.cfg['der_var'][self.cur_err_func]:
					self.output_grads[ef][key] = self.der[key][ef]
				
			self.output_vars = {} # for variable updating
			for ef in self.cfg['err_func']:
				self.output_vars[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.output_vars[ef][key] = self.vars[self.var_dict[key]]
			
			#### put the used derivative into a list, depending on the cfg
			grads_and_vars = {}
			for ef in self.cfg['err_func']:
				grads_and_vars[ef] = [
					der_pair[key] for key in self.cfg['der_var'][ef]
				]
			self.grads_and_vars = grads_and_vars

			#### optimize using AdagradOptimizer
			opt = {}
			self.train = {}
			for ef in self.cfg['err_func']:
				opt[ef] = tf.train.AdagradOptimizer(learning_rate=self.cfg['learn_rate'])
				self.train[ef] = opt[ef].apply_gradients(
					grads_and_vars[ef]
				)

			#save a reference for easy debugging purposes - there are some 
			#automatic ways of pulling this data out of the graph but this 
			#is much easier when prototyping
			self.vars['t'] = I_t
			self.vars['x'] = I_x
			self.vars['y'] = I_y
			self.vars['mu_f'] = mu_f
			self.vars['k'] = k
			self.vars['xx'] = I_xx
			self.vars['yy'] = I_yy
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

			# add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in),
				mu_s.assign(self.mu_s),
			)
			# assign final derivatives
			self.assign_der_f = {}
			for ef in self.cfg['err_func']:
				self.assign_der_f[ef] = {}
				for key in self.cfg['der_var'][self.cur_err_func]:
					self.assign_der_f[ef][key] = \
						self.der_f[key].assign(self.input_der[key])

			# assign modified variables
			self.assign_new_var = {}
			for ef in self.cfg['err_func']:
				self.assign_new_var[ef] = {}
				for key in self.cfg['der_var'][self.cur_err_func]:
					self.assign_new_var[ef][key] = \
						self.vars[self.var_dict[key]].assign(
							self.input_var[key]
						)

			#do not add anything to the compute graph after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	"""Here we define several error functions"""
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
			cols = int(self.cfg['szy_sensor'])
			rows = int(self.cfg['szx_sensor'])
		elif PADDING == 'VALID':
			rows_cut = int(
				(self.cfg['fxx'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['fyy'].shape[0]-1)/2
			)
			cols = int(self.cfg['szy_sensor'] - (self.cfg['wy'].shape[0]-1))
			rows = int(self.cfg['szx_sensor'] - (self.cfg['wx'].shape[1]-1))
		
		self.vars['Z_valid'] = self.vars['Z'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['U_valid'] = self.vars['U'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut,
			:
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
		cols = int(self.cfg['szy_sensor'])
		rows = int(self.cfg['szx_sensor'])
		self.vars['Z_gt_valid'] = self.vars['Z_gt'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['Z_gt_valid_flat'] = tf.reshape(
			self.vars['Z_gt_valid'], [-1]
		)
		return 

	def compute_err(self):
		self.valid_region()
		self.vars['Z_err_valid'] = tf.abs(
			self.vars['Z_valid'] - self.vars['Z_gt_valid']
		)
		self.vars['Z_err_valid_flat'] = tf.reshape(
			self.vars['Z_err_valid'],[-1]
		)
		return

	def compute_angle(self):
		# This function computes the angle from each point (Z_gt, Z_est) 
		# to the (mu_f,mu_f) point
		self.valid_region()
		stable_factor = 1e-8
		x_dir = self.cfg['mu_f'] - self.vars['Z_gt_valid'] + stable_factor
		y_dir = self.cfg['mu_f'] - self.vars['Z_valid'] + stable_factor
		self.vars['Z_err_valid'] = tf.atan(x_dir/y_dir)*180.0/np.pi-45
		self.vars['Z_err_valid_flat'] = tf.reshape(
			self.vars['Z_err_valid'],[-1]
		)
		return

	def tan_square_err(self):
		self.compute_angle()
		return tf.square(
			tf.tan(self.vars['Z_err_valid_flat'] / 180 * np.pi * 1.3)
		)

	def half_norm_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return tf.sqrt(self.vars['Z_err_valid_flat']+0.01)

	def one_norm_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return self.vars['Z_err_valid_flat']

	def one_norm_err1(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return self.vars['Z_err_valid_flat']

	def huber_err(self):
		# compute the huber loss, not useful for 
		# we want it to be in the same scale as the one norm
		delta = 0.05
		self.compute_err()
		def f1(): return tf.square(val)/2/delta
		def f2(): return val - delta/2
		temp = tf.unpack(self.vars['Z_err_valid_flat'])
		self.vars['loss_func'] = []
		for i in range(len(temp)):
			val = temp[i]
			self.vars['loss_func'].append(tf.cond(
					val < delta,
					f1,
					f2
				)
			) 		
		return tf.pack(self.vars['loss_func'])

	def two_norm_err(self):
		# Compute the one norm of the error for each pixel
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return self.vars['Z_err_valid_flat']**2

	def four_norm_err(self):
		# Compute the four norm of the error for each pixel
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return self.vars['Z_err_valid_flat']**4

	def sig_err(self):
		# Compute the sigmoid of the error for each pixel
		self.compute_err()
		delta = 0.03
		return 2/(
			1+tf.exp(-tf.square(self.vars['Z_err_valid_flat']/delta))
			)-1

	def inv_err(self):
		self.compute_err()
		delta = 0.01
		return -1/(tf.square(self.vars['Z_err_valid_flat']) + delta)

	"""training"""
	def update_loss(self):
		tmp_loss = np.sum(\
			self.session.run(
				self.vars['loss_func']\
				[self.cfg['err_func'][self.cur_err_idx]]
			)
		)
		self.loss += tmp_loss
		return

	def update_der(self):
		tmp_der = self.session.run(
			self.output_grads\
			[self.cfg['err_func'][self.cur_err_idx]]
		)
		# append the derivative to the list
		self.list_der.append(tmp_der)
		# sum over the derivatives
		for key in tmp_der.keys():
			if key not in self.ave_der:
				self.ave_der[key] = tmp_der[key]
			else:
				self.ave_der[key] += tmp_der[key]
		return 

	def finalize_der(self):
		# this function finalize the derivative and inputs 
		# it into the graph
		self.der_f_out = {}
		num_inst = len(self.list_der)
		# input the average derivative
		for key in self.ave_der.keys():
			self.der_f_out[self.input_der[key]] = \
				self.ave_der[key]/num_inst
		# 
		self.session.run(self.assign_der_f[self.cur_err_func], self.der_f_out)
		return

	def apply_der(self):
		# This function conduct SGD
		self.session.run(self.train)
		return

	def clear_der(self):
		# clear the past result
		self.list_der = []
		self.ave_der = {}
		self.loss = 0
		return

	def get_old_var(self):
		self.old_var = self.session.run(self.output_vars[self.cur_err_func])
		return

	def update_apply_var(self, step):
		# this function computes the new values of variables and
		# input it into the graph
		self.new_var = {}
		for key in self.old_var.keys():
			self.new_var[self.input_var[key]] = self.old_var[key] - \
				self.der_f_out[self.input_der[key]] * step
		self.session.run(self.assign_new_var[self.cur_err_func], self.new_var)
		return

	def print_loss(self):
		num_inst = len(self.list_der)
		tmp = self.loss/num_inst
		# clear the loss
		self.loss = 0
		print("Current average loss: ", tmp)
		return tmp

	def print_grads_and_vars(self):
		temp = self.session.run(self.grads_and_vars[self.cur_err_func])
		print("Current grads and vars: ", temp)
		return temp

	def heatmap(self, Z_flat, Z_gt_flat, fig, s1,s2,s3, fig_name):
		# return if no points given
		if Z_flat.shape[0] == 0:
			return
		# compute average error
		err = np.mean(np.abs(Z_flat - Z_gt_flat))

		# draw the heatmap of depth prediction
		step_in_m = 0.01
		min_depth = Z_gt_flat.min()-step_in_m/2
		max_depth = Z_gt_flat.max()+step_in_m/2
		dranges = [
			[min_depth, max_depth],
			[min_depth, max_depth]
		]
		bin_nums = [\
			int(np.rint((max_depth-min_depth)/step_in_m)),\
			int(np.rint((max_depth-min_depth)/step_in_m)),\
		]

		htmap, yedgs, xedgs = np.histogram2d(\
			Z_flat, \
			Z_gt_flat, \
			bins=bin_nums, \
			range=dranges, \
		)
		
		# normalize
		hist_gt, gtedgs = np.histogram(Z_gt_flat, bins=bin_nums[1], range=dranges[1])
		htmap /= hist_gt
		
		# since the batch is in equal size, we could directly use imshow
		title = fig_name+",err: "+str(err)
		ax = fig.add_subplot(s1,s2,s3, title=title)
		extent = [\
			min_depth, 
			max_depth, 
			min_depth, 
			max_depth
		]
		plt.imshow(htmap, interpolation='bilinear', origin='low', extent=extent)
		plt.colorbar()
		plt.xlabel('True depth (m)')
		plt.ylabel('Estimated depth (m)')
		plt.xlim((min_depth, max_depth))
		plt.ylim((min_depth, max_depth))
		plt.plot([min_depth, max_depth], [min_depth, max_depth], 'k')

		return

	def one_step_training_SGD(self, I, Loc, mu_s):
		# perform one step training by SGD
		print("Below is a new optimization step.")
		draw_list = np.empty((0,2),dtype = np.float32)
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, mu_s[i])

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# Query some results for drawing
			query_list = ['Z_valid_flat','Z_gt_valid_flat']
			res = self.query_results(query_list)
			# Update the derivative
			self.update_loss()
			self.update_der()
			draw_list = np.concatenate(
				(draw_list,
					np.column_stack(
						(
							res['Z_valid_flat'], 
							res['Z_gt_valid_flat']
						)
					)
				),axis = 0
			)
		print("Parameters before moving:")
		self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# conduct optimization
		self.apply_der()
		self.clear_der()

		# draw the training result
		min_depth = draw_list[:,1].min()
		max_depth = draw_list[:,1].max()
		plt.plot([min_depth, max_depth], [min_depth, max_depth])	
		plt.plot(draw_list[:,1],draw_list[:,0],'ro')
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.draw()
		return

	def one_round_loss(self, I, Loc, mu_s):
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, mu_s[i])

			# Update the loss
			self.update_loss()
		return self.print_loss()

	def one_step_training_force(self, I, Loc, mu_s, step, min_step):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss
		print("Below is a new optimization step")
		draw_list = np.empty((0,2),dtype = np.float32)
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, mu_s[i])

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# Query some results for drawing
			query_list = ['Z_valid_flat','Z_gt_valid_flat']
			res = self.query_results(query_list)
			# print(res['U_valid'])

			# Update the derivative
			self.update_loss()
			self.update_der()
			draw_list = np.concatenate(
				(draw_list,
					np.column_stack(
						(
							res['Z_valid_flat'], 
							res['Z_gt_valid_flat']
						)
					)
				),axis = 0
			)
		print("Parameters before moving:")
		old_loss = self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# draw the result before training
		plt.figure()
		min_depth = draw_list[:,1].min()
		max_depth = draw_list[:,1].max()
		plt.plot([min_depth, max_depth], [min_depth, max_depth])	
		plt.plot(draw_list[:,1],draw_list[:,0],'ro')
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.draw()

		# starting to find a good step size
		print("Start to find a good step size")
		self.get_old_var()
		while(step > min_step):
			print("Current step size:", step)
			self.update_apply_var(step)
			# run one round
			new_loss = self.one_round_loss(I, Loc, mu_s)
			if new_loss < old_loss:
				step = step * 2
				break
			step = step / 2
			
		# if the step grows too small
		# turn back to the old variables
		if step <= min_step:
			step = 0
			self.update_apply_var(step)

		self.clear_der()
		return step