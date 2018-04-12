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
from training_focalFlowNet import training_focalFlowNet
from training_focalFlowNet import KEY_RANGE
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os, glob

NAN_COLOR = np.array([0,0,255])
FONT_COLOR = (0,255,0)

PADDING = 'VALID'
class training_pyConfLensFlowNetFast(training_focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = None):
		#default configuraiton
		self.cfg = []
		self.cfg.append({
			# finite difference coefficient
			'gauss': np.array([[0.0023,0.0216,0.0216,0.0023],\
				[0.0216,0.2046,0.2046,0.0216],\
				[0.0216,0.2046,0.2046,0.0216],\
				[0.0023,0.0216,0.0216,0.0023]]),
			'ft': np.array([[[-0.5,0.5]]]),
			'fave': np.array([[[0.5,0.5]]]),
			'a0' : 1, #sensor distance
			'da0_ratio': 1, #adjust the scale to converge faster
			'a1' : 1, #focal distance
			'da1_ratio': 1, #adjust the scale to converge faster
			'Z_0': 0,
			'dZ_0_ratio': 1,
			# convolution window
			'separable' : True, # indicator of separability of conv
			'w' : np.ones((35,35)), # unseparable window
			'wx': np.ones((1,345)), # separable window
			'wy': np.ones((345,1)), # separable window
			# other parameters
			'szx_sensor': 200,
			'szy_sensor': 200,
			'outs': 'Z',
			'learn_rate': 0.01,
			'err_func': 'huber_err',
			'conf_func': 'baseline_conf',
			'fc': \
				np.array(\
					[\
						[[1,1],[1,1],[1,1]],\
						[[1,1],[1,1],[1,1]],\
						[[1,1],[1,1],[1,1]],\
					]\
				),
			'ac': 1,
		})

		# change configurations according to the input
		if cfg != None:
			self.cfg = cfg

		# add the cfg of fusion
		self.fused_cfg = {
			'separable'		:		True,
			'wx'			:		np.ones((1,3)),
			'wy'			:		np.ones((3,1)),
			'w'				:		np.ones((3,3)),
		}

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

		# variables
		self.vars = []
		self.vars_align = {}
		self.vars_fuse = {}

		# derivatives
		self.der = []
		self.der_pair = []
		self.input_der = []
		self.der_f = []
		self.input_var = []
		self.output_grads = []
		self.output_vars = []
		self.grads_and_vars = []
		self.input_data = []
		self.assign_der_f = []
		self.assign_new_var = []
		self.ave_der = \
			[{} for i in range(len(self.cfg))]
		self.list_der = \
			[[] for i in range(len(self.cfg))]
		# training
		self.loss = 0
		self.cur_err_idx = 0
		self.cur_err_func = self.cfg[0]['err_func'][0]
		
		# bi-directional mapping of basic derivatives
		self.der_dict = {
			'fave':'dLdfave',
			'ft':'dLdft',
			'a0':'dLda0',
			'a1':'dLda1',
			'Z_0':'dLdZ_0',
		}
		self.var_dict = {
			'dLdfave':'fave',
			'dLdft':'ft',
			'dLda0':'a0',
			'dLda1':'a1',
			'dLdZ_0': 'Z_0',
		}

		self.image_to_show = ['Z','conf','Z_gt']
		self.vars_to_fuse = ['Z','u_1','u_2','u_3','u_4','conf']

		# miscellaneous
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()
		
		self.netName = "Training Lens Flow"

	"""describes the computations (graph) to run later
		-make all algorithmic changes here
		-note that tensorflow has a lot of support for backpropagation 
		 gradient descent/training, so you can build another part of the 
		 graph that computes and updates weights here as well. 
	"""
	def input_images(self, I, Z):
		# import data into the network
		# I, I_lap_batch and Z should be a n-element tuple
		input_dict = {}
		input_dict[self.I_in] = I
		input_dict[self.Z_in] = Z
		self.session.run(self.input_data, input_dict)
		return

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

			Z_init = np.zeros(
				self.resolution[0],
				dtype = np.float32
			)
			self.Z_in = tf.Variable(Z_init)
			Z_gt = tf.Variable(Z_init)
			Z_gt0 = []

			Z_0 = []
			a0 = []
			a1 = []
			gauss = []
			ft = []
			fave = []
			if self.cfg[0]['separable']:
				wx = []
				wy = []
			else:
				w = []

			I_t = []
			I_lap = []

			u_1wt = []
			u_2wt = []
			u_3wt = []
			u_4wt = []
			u_1w = []
			u_2w = []
			u_3w = []
			u_4w = []
			Zw = []
			uncw = []
			confw = []

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
				Z_0.append(tf.Variable(self.cfg[i]['Z_0'], dtype = tf.float32))
				
				a0.append(tf.Variable(self.cfg[i]['a0'], dtype = tf.float32))
				a1.append(tf.Variable(self.cfg[i]['a1'], dtype = tf.float32))

				gauss.append(tf.Variable(self.cfg[i]['gauss'], dtype = tf.float32))
				ft.append(tf.Variable(self.cfg[i]['ft'], dtype = tf.float32))
				fave.append(tf.Variable(self.cfg[i]['fave'], dtype = tf.float32))

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

				if self.cfg[0]['separable']:
					wx.append(tf.Variable(
						self.cfg[i]['wx'], dtype = tf.float32
					))
					wy.append(tf.Variable(
						self.cfg[i]['wy'], dtype = tf.float32
					))
				else:
					w.append(tf.Variable(
						self.cfg[i]['w'], dtype = tf.float32
					))

				# Generate the differential images
				I_t.append(dIdt(I_batch[i], ft[i]))
				I_lap.append(dIdt(I_lap_batch[i], fave[i]))

				# windowed version
				if self.cfg[0]['separable']:
					u_1wt.append(separable_window(
						I_lap[i]*I_lap[i], 
						wx[i], 
						wy[i], 
						PADDING,
					))
					u_2wt.append(separable_window(
						I_t[i]*I_lap[i], 
						wx[i], 
						wy[i], 
						PADDING,
					))
				else:
					u_1wt.append(unseparable_window(
						I_lap[i]*I_lap[i], 
						w[i], 
						PADDING,
					))
					u_2wt.append(unseparable_window(
						I_t[i]*I_lap[i], 
						w[i],
						PADDING,
					))

				u_1w.append(tf.squeeze(u_1wt[i], [0,3]))
				u_2w.append(tf.squeeze(u_2wt[i], [0,3]))
				u_3w.append(a0[i] * a1[0] * u_1w[i])
				u_4w.append(-u_2w[i] + a0[i] * u_1w[i])
				Zw.append((u_3w[i]*u_4w[i]) / (u_4w[i]*u_4w[i] + 1e-5) + Z_0[0])

				# unwindowed version
				u_1.append(I_lap[i])
				u_2.append(I_t[i])
				u_3.append(a0[i] * a1[0] * u_1[i])
				u_4.append(-u_2[i] + a0[i] * u_1[i])
				Z.append((u_3[i]*u_4[i]) / (u_4[i]*u_4[i] + 1e-5) + Z_0[0])
				
				#save references to required I/O]
				self.vars.append({})
				self.vars[i]['I'] = I_batch[i]

				# windowed version
				self.vars[i]['u_1w'] = u_1w[i]
				self.vars[i]['u_2w'] = u_2w[i]
				self.vars[i]['u_3w'] = u_3w[i]
				self.vars[i]['u_4w'] = u_4w[i]
				self.vars[i]['Zw'] = Zw[i]
				
				# unwindowed version
				self.vars[i]['u_1'] = u_1[i]
				self.vars[i]['u_2'] = u_2[i]
				self.vars[i]['u_3'] = u_3[i]
				self.vars[i]['u_4'] = u_4[i]
				self.vars[i]['Z'] = Z[i]

				#save a reference for easy debugging purposes - there are some 
				#automatic ways of pulling this data out of the graph but this 
				#is much easier when prototyping
				self.vars[i]['I_t'] = I_t[i]
				self.vars[i]['I_lap'] = I_lap[i]
				self.vars[i]['a0'] = a0[i]
				self.vars[i]['a1'] = a1[i]

			# align depth and confidence maps
			self.align_maps(['u_1','u_2','u_3','u_4'])

			# compute the aligned version of Z
			self.vars_align['Z'] = \
				self.vars_align['u_3']*self.vars_align['u_4'] / \
				(self.vars_align['u_4']*self.vars_align['u_4'] + 1e-5)
			Z_align = []
			for i in range(len(self.cfg)):
				Z_align.append(self.vars_align['Z'][:,:,i] + Z_0[0])
				self.vars[i]['Z_align'] = Z_align[i]
			self.vars_align['Z'] = tf.pack(Z_align, 2)

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[i]['conf_func']+'()')

			# cut the valid region for windowed version
			self.valid_windowed_region()

			# save the ground truth
			self.vars_fuse['Z_gt'] = Z_gt
			Z_gt_tmp = [Z_gt for i in range(len(self.cfg))]
			self.vars_align['Z_gt'] = tf.pack(Z_gt_tmp, 2)

			# fusion
			self.softmax_fusion()	

			# compute windowed version of fused depth
			# shorten the name
			separable = self.fused_cfg['separable']
			wx = self.fused_cfg['wx'].astype(np.float32)
			wy = self.fused_cfg['wy'].astype(np.float32)
			w = self.fused_cfg['w'].astype(np.float32)

			u_1 = self.vars_fuse['u_1']
			u_2 = self.vars_fuse['u_2']
			u_3 = self.vars_fuse['u_3']
			u_4 = self.vars_fuse['u_4']
			conf = self.vars_fuse['conf']
			if separable:
				u_3w = tf.squeeze(separable_window(
					u_1*u_3, 
					wx, 
					wy, 
					PADDING,
				), [0,3])
				u_4w = tf.squeeze(separable_window(
					u_1*u_4, 
					wx, 
					wy, 
					PADDING,
				), [0,3])
				confw = tf.squeeze(separable_window(
					conf,
					wx,
					wy,
					PADDING
					)/np.sum(wx)/np.sum(wy),
				[0,3])
			else:
				u_3w = tf.squeeze(unseparable_window(
					u_1*u_3, 
					w, 
					PADDING,
				), [0,3])
				u_4w = tf.squeeze(unseparable_window(
					u_1*u_4, 
					w,
					PADDING,
				), [0,3])
				confw = 1 - tf.squeeze(separable_window(
					1-conf,
					w,
					PADDING
					)/np.sum(w),
				[0,3])

			Zw = u_3w / u_4w + Z_0[0] # assume Z_0 to be the same for all res
			self.vars_fuse['Zw'] = Zw
			self.vars_fuse['confw'] = confw
			self.valid_windowed_region_fuse()

			# compute error
			self.vars_fuse['loss_func'] = {}
			for ef in self.cfg[0]['err_func']:
				self.vars_fuse['loss_func'][ef] = \
					eval('self.'+ef+'()')		

			# compute derivatives for prediction
			for i in range(len(self.cfg)):
				self.der.append({})
				self.der[i]['dLdft'] = {}
				self.vars[i]['ft'] = ft[i]			

				self.der[i]['dLdfave'] = {}
				self.vars[i]['fave'] = fave[i]

				self.der[i]['dLda0'] = {}
				self.der[i]['dLda1'] = {}
				self.der[i]['dLdZ_0'] = {}
				self.vars[i]['a0'] = a0[i]
				self.vars[i]['a1'] = a1[i]
				self.vars[i]['Z_0'] = Z_0[i]

				# self.vars[i]['dZda0'], = tf.gradients(
				# 	Z_align[i],  [a0[i]]
				# )

				# self.vars[i]['dZ_fuseda0'], = tf.gradients(
				# 	self.vars_fuse['Z'], [a0[i]]
				# )

				# self.vars[i]['dZ_fusedZ'], = tf.gradients(
				# 	self.vars_fuse['Z'], [Z_align[i]]
				# )

				# self.vars[i]['dZ_fusedconf'], = tf.gradients(
				# 	self.vars_fuse['Z'], [self.vars[i]['conf_align']]
				# )

				# self.vars[i]['dLda0'], = tf.gradients(
				# 	self.vars_fuse['loss_func'][self.cfg[0]['err_func'][0]],
				# 	[a0[i]]
				# )

				# self.vars[i]['dconfda0'], = tf.gradients(
				# 	self.vars[i]['conf_align'], [a0[i]]
				# )

				# self.vars[i]['dwsdconf'], = tf.gradients(
				# 	self.vars[i]['ws_align'], [self.vars[i]['conf_align']]
				# )

				# self.vars[i]['dwsda0'], = tf.gradients(
				# 	self.vars[i]['ws_align'], [a0[i]]
				# )

				# self.vars_fuse['dZ_fusedws'], = tf.gradients(
				# 	self.vars_fuse['Z'], [self.vars_fuse['ws']]
				# )

				# self.vars_fuse['dZ_fusedZ'], = tf.gradients(
				# 	self.vars_fuse['Z'], [self.vars_align['Z']]
				# )

				for ef in self.cfg[0]['err_func']:
					dLdft, dLdfave = \
						tf.gradients(\
							self.vars_fuse['loss_func'][ef],\
							[ft[i], fave[i]]
						)

					if dLdft == None:
						self.der[i]['dLdft'][ef] = tf.zeros(tf.shape(self.vars[i]['ft']))
					else:
						# restrict the symmetry:
						# dLdft + tf.reverse(dLdft) = 0
						dLdft = dLdft - tf.reverse(
							dLdft, dims = [False, False, True]
						)
						dLdft = dLdft\
							+ tf.reverse(dLdft, dims = [True, False, False])\
						 	+ tf.reverse(dLdft, dims = [False, True, False])\
						 	+ tf.reverse(dLdft, dims = [True, True, False])
						dLdft = dLdft\
							+ tf.transpose(dLdft, perm = [1,0,2])
						dLdft = dLdft\
							+ tf.reverse(dLdft, dims = [True, False, False])
						dLdft = dLdft\
							+ tf.reverse(dLdft, dims = [False, True, False])
						dLdft = dLdft / 64.
						self.der[i]['dLdft'][ef] = dLdft
					
					if dLdfave == None:
						self.der[i]['dLdfave'][ef] = tf.zeros(tf.shape(self.vars[i]['fave']))
					else:
						# derivative w.r.t. to fave
						dLdfave = dLdfave
						self.der[i]['dLdfave'][ef] = dLdfave

					#### derivative w.r.t. optical parameters
					dLda0, dLda1, dLdZ_0 = tf.gradients(
						self.vars_fuse['loss_func'][ef],\
						[a0[i], a1[i], Z_0[i]]
					)
					if dLda0 == None:
						dLda0 = tf.constant(0,dtype =tf.float32)
					if dLda1 == None:
						dLda1 = tf.constant(0,dtype =tf.float32)
					if dLdZ_0 == None:
						dLdZ_0 = tf.constant(0,dtype =tf.float32)

					dLda0 = dLda0 * self.cfg[i]['da0_ratio']
					dLda1 = dLda1 * self.cfg[i]['da1_ratio']
					dLdZ_0= dLdZ_0 * self.cfg[i]['dZ_0_ratio']
					self.der[i]['dLda0'][ef] = dLda0
					self.der[i]['dLda1'][ef] = dLda1
					self.der[i]['dLdZ_0'][ef] = dLdZ_0

					# #### for debugging
					# dLdZ = tf.gradients(
					# 	self.vars_fuse['loss_func'][ef],\
					# 	self.vars[i]['Z'],
					# )
					# self.vars[i]['dLdZ'] = dLdZ

					# dconfdconf = tf.gradients(
					# 	self.vars_fuse['conf'],\
					# 	self.vars[i]['conf'],
					# )
					# self.vars[i]['dconfdconf'] = dconfdconf

					# dconfda0, dconfda1, dconfdZ = tf.gradients(
					# 	self.vars[i]['conf'],\
					# 	[self.vars[i]['a0'],self.vars[i]['a1'], self.vars[i]['Z']]
					# )
					# self.vars[i]['dconfda0'] = dconfda0
					# self.vars[i]['dconfda1'] = dconfda1
					# self.vars[i]['dconfdZ'] = dconfdZ

					# dZda0, dZda1 = tf.gradients(
					# 	self.vars[i]['Z'],\
					# 	[self.vars[i]['a0'],self.vars[i]['a1']]
					# )
					# self.vars[i]['dZda0'] = dZda0
					# self.vars[i]['dZda1'] = dZda1


			# compute derivatives for uncertainty
			eval('self.'+self.cfg[i]['conf_func']+'_der()')
			
			# fit the derivative into proper format
			# and for inputting final derivative
			for i in range(len(self.cfg)):
				self.der_pair.append({})
				self.input_der.append({})
				self.der_f.append({})
				self.input_var.append({})
				for key in self.der_dict.keys():
					self.input_der[i][self.der_dict[key]] = \
						tf.Variable(self.cfg[i][key], \
							dtype = tf.float32
						)
					self.der_f[i][self.der_dict[key]] = \
						tf.Variable(self.cfg[i][key], \
							dtype = tf.float32
						)
					self.der_pair[i][self.der_dict[key]] = (
						#turn it into a tensor
						self.der_f[i][self.der_dict[key]] + 0.
						, self.vars[i][key] 
					)
					self.input_var[i][self.der_dict[key]] = \
						tf.Variable(self.cfg[i][key], \
							dtype = tf.float32
						)

				self.output_grads.append({})
				for ef in self.cfg[0]['err_func']:
					self.output_grads[i][ef] = {}
					for key in self.cfg[0]['der_var'][ef]:
						self.output_grads[i][ef][key] =\
							self.der[i][key][ef]
					
				self.output_vars.append({}) # for variable updating
				for ef in self.cfg[0]['err_func']:
					self.output_vars[i][ef] = {}
					for key in self.cfg[0]['der_var'][ef]:
						self.output_vars[i][ef][key] =\
							self.vars[i][self.var_dict[key]]
					
				#### put the used derivative into a list, depending on the cfg
				self.grads_and_vars.append({})
				for ef in self.cfg[0]['err_func']:
					self.grads_and_vars[i][ef] = [
						self.der_pair[i][key] \
						for key in self.cfg[0]['der_var'][ef]
					]

				# assign final derivatives			
				self.assign_der_f.append({})
				for ef in self.cfg[0]['err_func']:
					self.assign_der_f[i][ef] = {}
					for key in self.cfg[0]['der_var'][ef]:
						self.assign_der_f[i][ef][key] = \
							self.der_f[i][key].assign(\
								self.input_der[i][key]\
							)

				# assign modified variables
				self.assign_new_var.append({})
				for ef in self.cfg[0]['err_func']:
					self.assign_new_var[i][ef] = {}
					for key in self.cfg[0]['der_var'][ef]:
						self.assign_new_var[i][ef][key] = \
							self.vars[i][self.var_dict[key]].assign(
								self.input_var[i][key]
							)

			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			self.input_data = tf.group(\
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in)
			)

			#do not add anything to the compute graph 
			#after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	# confidence functions
	def outlier_nonlinearity(self):
		# this function set confidence of prediction outside 
		# the working range to zero using sigmoid
		d = 0.1 # half width of soft margin
		k = 1/d
		for i in range(len(self.cfg)):
			lo,hi = self.cfg[i]['wr']
			x = self.vars[i]['conf']
			Z = self.vars[i]['Z']
			# we use arctan function to restrict the exponential
			# from being too large
			conf =  1/(tf.exp(k*(tf.atan(Z-hi-d)))+1) * \
					1/(tf.exp(k*(tf.atan(lo-d-Z)))+1) * x
			self.vars[i]['conf'] = conf

		return

	def baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability
		# the windowed flag indicates whether we compute windowed
		for i in range(len(self.cfg)):
			# windowed version
			self.vars[i]['confw'] = \
				(self.vars[i]['u_4w']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3w']**2 + \
					self.vars[i]['u_4w']**2 + \
					self.vars[i]['u_4w']**4 + \
					1e-10\
				)


			# unwindowed version
			self.vars[i]['conf'] = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3']**2 + \
					self.vars[i]['u_4']**2 + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				)

		# aligned version
		self.align_maps(['conf'])

		return 

	def baseline_conf_der(self):
		return

	def w_baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability
		# the windowed flag indicates whether we compute windowed
		w_bc = [] # the weight of baseline confidence for each layer

		for i in range(len(self.cfg)):
			# weights
			w_bc.append(\
				tf.Variable(
					self.cfg[i]['w_bc'],
					dtype = tf.float32
				)	
			)

			# windowed version
			self.vars[i]['confw'] = \
				(self.vars[i]['u_4w']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3w']**2 + \
					self.vars[i]['u_4w']**2 + \
					self.vars[i]['u_4w']**4 + \
					1e-10\
				) * w_bc[i]


			# unwindowed version
			self.vars[i]['conf'] = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3']**2 + \
					self.vars[i]['u_4']**2 + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				) * w_bc[i]
			self.vars[i]['w_bc'] = w_bc[i]


		# aligned version
		# self.align_maps(['conf'])
		conf_align = []
		tmp_align = \
			(self.vars_align['u_4']**2 + 1e-20)/\
			tf.sqrt(\
				self.vars_align['u_3']**2 + \
				self.vars_align['u_4']**2 + \
				self.vars_align['u_4']**4 + \
				1e-10\
			)

		for i in range(len(self.cfg)):
			conf_align.append(
				tmp_align[:,:,i] * w_bc[i]
			)
			self.vars[i]['conf_align'] = conf_align[i]
		self.vars_align['conf'] = tf.pack(conf_align, 2)

		return 

	def w_baseline_conf_der(self):
		# setting up derivatives for the confidence
		vars_to_add = ['w_bc']
		
		return self.generate_conf_der(vars_to_add)

	def wr_baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability,
		# use the working range to cut the confidence
		# and use weight for each layer
		# the windowed flag indicates whether we compute windowed
		w_bc = [] # the weight of baseline confidence for each layer
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


			# windowed version
			confw_tmp = \
				(self.vars[i]['u_4w']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3w']**2 + \
					self.vars[i]['u_4w']**2 + \
					self.vars[i]['u_4w']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Zw = self.vars[i]['Zw']
			cutw = tf.sigmoid(-k*(Zw-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Zw))

			# final confidence
			self.vars[i]['confw'] = confw_tmp * w_bc[i] * cutw


			# unwindowed version
			conf_tmp = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					self.vars[i]['u_3']**2 + \
					self.vars[i]['u_4']**2 + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Z = self.vars[i]['Z']
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))

			# final confidence
			self.vars[i]['conf'] = conf_tmp * w_bc[i] * cut
			self.vars[i]['cut'] = cut
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['lo'] = lo[i]
			self.vars[i]['hi'] = hi[i]


		# aligned version
		# self.align_maps(['conf'])
		conf_align = []
		tmp_align = (self.vars_align['u_4']**2 + 1e-20)/\
			tf.sqrt(\
				self.vars_align['u_3']**2 + \
				self.vars_align['u_4']**2 + \
				self.vars_align['u_4']**4 + \
				1e-10\
			)

		for i in range(len(self.cfg)):
			Z = self.vars_align['Z'][:,:,i]
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))
			# aligned confidence
			conf_align.append(
				tmp_align[:,:,i] * w_bc[i] * cut
			)
			self.vars[i]['conf_align'] = conf_align[i]
		self.vars_align['conf'] = tf.pack(conf_align,2)

		return

	def wr_baseline_conf_der(self):
		# setting up derivatives for the confidence
		vars_to_add = ['w_bc', 'lo', 'hi']
		
		return self.generate_conf_der(vars_to_add)

	def w2r_baseline_conf(self):
		# this function computes the confidence and 
		# uncertainty according to stability,
		# use the working range to cut the confidence
		# and use weight for each layer
		# the windowed flag indicates whether we compute windowed
		w_bc = [] # the weight of baseline confidence for each layer
		w_bc1 = []
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


			# windowed version
			confw_tmp = \
				(self.vars[i]['u_4w']**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars[i]['u_3w']**2 + \
					w_bc1[i] * self.vars[i]['u_4w']**2 + \
					self.vars[i]['u_4w']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Zw = self.vars[i]['Zw']
			cutw = tf.sigmoid(-k*(Zw-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Zw))

			# final confidence
			self.vars[i]['confw'] = confw_tmp * cutw


			# unwindowed version
			conf_tmp = \
				(self.vars[i]['u_4']**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars[i]['u_3']**2 + \
					w_bc1[i] * self.vars[i]['u_4']**2 + \
					self.vars[i]['u_4']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Z = self.vars[i]['Z']
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))

			# final confidence
			self.vars[i]['conf'] = conf_tmp * cut
			self.vars[i]['cut'] = cut
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['w_bc1'] = w_bc1[i]
			self.vars[i]['lo'] = lo[i]
			self.vars[i]['hi'] = hi[i]


		# aligned version
		# self.align_maps(['conf'])
		conf_align = []

		for i in range(len(self.cfg)):
			tmp_align = (self.vars_align['u_4'][:,:,i]**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars_align['u_3'][:,:,i]**2 + \
					w_bc1[i] * self.vars_align['u_4'][:,:,i]**2 + \
					self.vars_align['u_4'][:,:,i]**4 + \
					1e-10\
				)	
			Z = self.vars_align['Z'][:,:,i]
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))
			# aligned confidence
			conf_align.append(
				tmp_align * cut
			)
			self.vars[i]['conf_align'] = conf_align[i]

		self.vars_align['conf'] = tf.pack(conf_align,2)

		return

	def w2r_baseline_conf_der(self):
		# setting up derivatives for the confidence
		vars_to_add = ['w_bc', 'w_bc1', 'lo', 'hi']
		
		return self.generate_conf_der(vars_to_add)

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


			# windowed version
			confw_tmp = \
				(self.vars[i]['u_4w']**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars[i]['u_3w']**2 + \
					w_bc1[i] * self.vars[i]['u_4w']**2 + \
					w_bc2[i] * self.vars[i]['u_3w']*self.vars[i]['u_4w'] + \
					self.vars[i]['u_4w']**4 + \
					1e-10\
				)

			# sigmoid cutting
			Zw = self.vars[i]['Zw']
			cutw = tf.sigmoid(-k*(Zw-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Zw))

			# final confidence
			self.vars[i]['confw'] = confw_tmp * cutw


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
			self.vars[i]['cut'] = cut
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['w_bc1'] = w_bc1[i]
			self.vars[i]['w_bc2'] = w_bc2[i]
			self.vars[i]['lo'] = lo[i]
			self.vars[i]['hi'] = hi[i]


		# aligned version
		# self.align_maps(['conf'])
		conf_align = []

		for i in range(len(self.cfg)):
			tmp_align = (self.vars_align['u_4'][:,:,i]**2 + 1e-20)/\
				tf.sqrt(\
					w_bc[i] * self.vars_align['u_3'][:,:,i]**2 + \
					w_bc1[i] * self.vars_align['u_4'][:,:,i]**2 + \
					w_bc2[i] * self.vars_align['u_3'][:,:,i]*self.vars_align['u_4'][:,:,i] + \
					self.vars_align['u_4'][:,:,i]**4 + \
					1e-10\
				)	
			Z = self.vars_align['Z'][:,:,i]
			cut = tf.sigmoid(-k*(Z-hi[i])) * \
				tf.sigmoid(-k*(lo[i]-Z))
			# aligned confidence
			conf_align.append(
				tmp_align * cut
			)
			self.vars[i]['conf_align'] = conf_align[i]

		self.vars_align['conf'] = tf.pack(conf_align,2)

		return

	def w3r_baseline_conf_der(self):
		# setting up derivatives for the confidence
		vars_to_add = ['w_bc', 'w_bc1', 'w_bc2', 'lo', 'hi']
		
		return self.generate_conf_der(vars_to_add)

	def generate_conf_der(self, vars_to_add):
		for var in vars_to_add:
			self.der_dict[var] = 'dLd'+var
			self.var_dict['dLd'+var] = var
			
		# shorten the notation and compute the derivative
		for i in range(len(self.cfg)):
			for var in vars_to_add:
				self.der[i]['dLd'+var] = {}

			for ef in self.cfg[0]['err_func']:
				for var in vars_to_add:
					self.der[i]['dLd'+var][ef], = tf.gradients(\
							self.vars_fuse['loss_func'][ef],\
							[self.vars[i][var],]
						)
					if self.der[i]['dLd'+var][ef] == None:
						self.der[i]['dLd'+var][ef] = \
							tf.zeros(tf.shape(self.vars[i][var]))
		return

	def posterior_conf(self):
		# this confidence function computes P(Z=Z_est | I)
		# with the Gaussian noise model: I = I_0 + e

		# the parameters are identical throughout all scales
		# so we only use the one for the first layer
		# original image noise variance
		C_e0 = tf.Variable(self.cfg[0]['C_e0'])

		# noise reduction ratio through layer
		e_ratio = tf.Variable(self.cfg[0]['e_ratio'], dtype=tf.float32)

		# \sum(f_t)^2 and \sum(nabla^2)^2
		ft_ssum = tf.Variable(self.cfg[0]['ft_ssum'], dtype=tf.float32)
		n2_ssum = tf.Variable(self.cfg[0]['n2_ssum'], dtype=tf.float32)

		#### UNWINDOWED VERSION
		# initializtion of some variables
		rho = []
		mu_3 = []
		mu_4 = []
		C_e = []
		C_3 = []
		C_4 = []
		Z = []
		
		# initialize the distribution
		dist = tf.contrib.distributions.Normal(mu=0., sigma=1.)

		for i in range(len(self.cfg)):
			# compute correlation factor rho
			a0 = self.vars[i]['a0']
			a1 = self.vars[i]['a1']
			rho_i  = 1/tf.sqrt(ft_ssum/n2_ssum/a0**2+1)
			rho.append(rho_i)

			# mean and variance for u_3 and u_4
			mu_3.append(self.vars[i]['u_3'])
			mu_4.append(self.vars[i]['u_4'])
			C_e.append(C_e0 * e_ratio**i)
			C_3.append(n2_ssum * a0**2 * a1**2 * C_e[i])
			C_4.append((ft_ssum + a0**2 * n2_ssum) * C_e[i])

			# Z
			Z.append(self.vars[i]['Z'])

			# intermediate variables
			a = tf.sqrt(
				(
					Z[i]**2/C_3[i] - \
					2*rho[i]*Z[i]/tf.sqrt(C_3[i]*C_4[i]) + \
					1/C_4[i]
				)
			)

			b = mu_3[i] * Z[i] / C_3[i] - \
				rho[i] * (mu_3[i] + mu_4[i]*Z[i])/tf.sqrt(C_3[i]*C_4[i]) + \
				mu_4[i] / C_4[i]

			c = mu_3[i]**2 / C_3[i] - \
				2*rho[i]*mu_3[i]*mu_4[i]/tf.sqrt(C_3[i]*C_4[i]) + \
				mu_4[i]**2 / C_4[i]

			d = tf.exp(
				(b**2 - c * a**2)/\
				(2 * (1 - rho[i]**2) * a**2)
			)

			coeff1 = b*d/tf.sqrt(2*np.pi)/tf.sqrt(C_3[i]*C_4[i])/a**3
			coeff2 = dist.cdf(b/tf.sqrt(1-rho[i]**2)/a) - \
				dist.cdf(-b/tf.sqrt(1-rho[i]**2)/a)
			coeff3 = tf.sqrt(1-rho[i]**2)/np.pi/tf.sqrt(C_3[i]*C_4[i])/a**2
			coeff4 = tf.exp(-c/2/(1-rho[i]**2))

			self.vars[i]['conf'] = coeff1*coeff2 + coeff3*coeff4
			self.vars[i]['C_3'] = C_3[i]
			self.vars[i]['C_4'] = C_4[i]
			self.vars[i]['rho'] = rho[i]
			self.vars[i]['a'] = a
			self.vars[i]['b'] = b
			self.vars[i]['c'] = c
			self.vars[i]['d'] = d
			self.vars[i]['C_e0'] = C_e0
			self.vars[i]['e_ratio'] = e_ratio
			self.vars[i]['ft_ssum'] = ft_ssum
			self.vars[i]['n2_ssum'] = n2_ssum


		#### WINDOWED VERSION
		# initializtion of some variables
		rho = []
		mu_3 = []
		mu_4 = []
		C_e = []
		C_3 = []
		C_4 = []
		Z = []
		
		# initialize the distribution
		dist = tf.contrib.distributions.Normal(mu=0., sigma=1.)

		for i in range(len(self.cfg)):
			# compute correlation factor rho
			a0 = self.vars[i]['a0']
			a1 = self.vars[i]['a1']
			rho_i  = 1/tf.sqrt(ft_ssum/n2_ssum/a0**2+1)
			rho.append(rho_i)

			# mean and variance for u_3 and u_4
			mu_3.append(self.vars[i]['u_3w'])
			mu_4.append(self.vars[i]['u_4w'])
			C_e.append(C_e0 * e_ratio**i)
			C_3.append(n2_ssum * a0**2 * a1**2 * C_e[i])
			C_4.append((ft_ssum + a0**2 * n2_ssum) * C_e[i])

			# Z
			Z.append(self.vars[i]['Zw'])

			# intermediate variables
			a = tf.sqrt(
				(
					Z[i]**2/C_3[i] - \
					2*rho[i]*Z[i]/tf.sqrt(C_3[i]*C_4[i]) + \
					1/C_4[i]
				)
			)

			b = mu_3[i] * Z[i] / C_3[i] - \
				rho[i] * (mu_3[i] + mu_4[i]*Z[i])/tf.sqrt(C_3[i]*C_4[i]) + \
				mu_4[i] / C_4[i]

			c = mu_3[i]**2 / C_3[i] - \
				2*rho[i]*mu_3[i]*mu_4[i]/tf.sqrt(C_3[i]*C_4[i]) + \
				mu_4[i]**2 / C_4[i]

			d = tf.exp(
				(b**2 - c * a**2)/\
				(2 * (1 - rho[i]**2) * a**2)
			)

			coeff1 = b*d/tf.sqrt(2*np.pi)/tf.sqrt(C_3[i]*C_4[i])/a**3
			coeff2 = dist.cdf(b/tf.sqrt(1-rho[i]**2)/a) - \
				dist.cdf(-b/tf.sqrt(1-rho[i]**2)/a)
			coeff3 = tf.sqrt(1-rho[i]**2)/np.pi/tf.sqrt(C_3[i]*C_4[i])/a**2
			coeff4 = tf.exp(-c/2/(1-rho[i]**2))

			self.vars[i]['confw'] = coeff1*coeff2 + coeff3*coeff4

		### NEED AN ALIGNED VERSION OR AN ALIGNMENT OF CONF
		self.align_maps(['conf'])

		return 

	def posterior_conf_der(self):
		# setting up derivatives for the confidence
		self.der_dict['C_e0'] = 'dLdC_e0'
		self.der_dict['e_ratio'] = 'dLde_ratio'
		self.der_dict['ft_ssum'] = 'dLdft_ssum'
		self.der_dict['n2_ssum'] = 'dLdn2_ssum'
		self.var_dict['dLdC_e0'] = 'C_e0'
		self.var_dict['dLde_ratio'] = 'e_ratio'
		self.var_dict['dLdft_ssum'] = 'ft_ssum'
		self.var_dict['dLdn2_ssum'] = 'n2_ssum'

		# shorten the notation		
		for i in range(len(self.cfg)):
			Ce0 = self.vars[i]['C_e0']
			e_ratio = self.vars[i]['e_ratio']
			ft_ssum = self.vars[i]['ft_ssum']
			n2_ssum = self.vars[i]['n2_ssum']

			self.der[i]['dLdC_e0'] = {}
			self.der[i]['dLde_ratio'] = {}
			self.der[i]['dLdft_ssum'] = {}
			self.der[i]['dLdn2_ssum'] = {}

			for ef in self.cfg[0]['err_func']:
				dLdCe0, dLde_ratio, dLdft_ssum, dLdn2_ssum = \
					tf.gradients(
						self.vars_fuse['loss_func'][ef],\
						[Ce0, e_ratio, ft_ssum, n2_ssum]
					)

				num_inst = tf.to_float(
						tf.shape(self.vars_fuse['loss_func'][ef])[0]
					)

				self.der[i]['dLdC_e0'][ef] = dLdCe0 / num_inst
				self.der[i]['dLde_ratio'][ef] = dLde_ratio / num_inst
				self.der[i]['dLdft_ssum'][ef] = dLdft_ssum / num_inst
				self.der[i]['dLdn2_ssum'][ef] = dLdn2_ssum / num_inst

		return 

	def filter_conf(self):
		# compute the confidence by using a filter 
		# and a nonlinearity function
		fc = []
		conf_tmp = []
		ac = []
		conf = []
		for i in range(len(self.cfg)):
			# convolve the images with filter
			fc.append(tf.Variable(\
				self.cfg[i]['fc'], dtype=tf.float32\
			))
			conf_tmp.append(dIdt(self.vars[i]['I'], fc))
			
			# nonlinearity
			ac.append(\
				tf.Variable(\
					self.cfg[i]['fc'],\
					dtype=tf.float32\
				)\
			)
			conf.append(conf_tmp[i] * ac[i])

			# save the variables
			self.vars[i]['conf'] = conf[i]
			self.vars[i]['fc'] = fc[i]
			self.vars[i]['ac'] = ac[i]

		self.align_maps(['conf'])

		return

	def filter_conf_der(self):
		# setting up derivatives for the variables
		self.der_dict['fc'] = 'dLdfc'
		self.der_dict['ac'] = 'dLdac'
		self.var_dict['dLdfc'] = 'fc'
		self.var_dict['dLdac'] = 'ac'
		for i in range(len(self.cfg)):
			self.der[i]['dLdfc'] = {}
			self.der[i]['dLdac'] = {}
			for ef in self.cfg[0]['err_func']:
				dLdfc, dLdac = \
					tf.gradients(
						self.vars_fuse['loss_func'][ef],\
						[fc[i], ac[i]]
					)
				num_inst = tf.to_float(
						tf.shape(self.vars_fuse['loss_func'][ef])[0]
					)
				# force dLdfc to be symmetric
				# and the derivative sum to zero
				dLdfc = dLdfc - tf.reverse(dLdfc, dims = [False, False, True])
				dLdfc = dLdfc + tf.reverse(dLdfc, dims = [True, False, False])\
				 	+ tf.reverse(dLdfc, dims = [False, True, False])\
				 	+ tf.reverse(dLdfc, dims = [True, True, False])
				dLdfc = dLdfc + tf.transpose(dLdfc, perm = [1,0,2])
				dLdfc = dLdfc + tf.reverse(dLdfc, dims = [True, False, False])
				dLdfc = dLdfc + tf.reverse(dLdfc, dims = [False, True, False])
				self.der[i]['dLdfc'][ef] = dLdfc / 64. / num_inst

				self.der[i]['dLdac'][ef] = dLdac / num_inst
		return

	# fusion methods
	def softmax_fusion(self):
		# reshape for softmax
		conf_flat = tf.reshape(
			self.vars_align['conf'],
			[-1, len(self.cfg)]
		)
		
		# not sure if it will cause numerical problem
		ws = tf.reshape(
			tf.nn.softmax(conf_flat*1e10),
			self.resolution[0]+(-1,)
		)
		# self.vars_fuse['ws'] = ws
		# for i in range(len(self.cfg)):
		# 	self.vars[i]['ws_align'] = ws[:,:,i]

		# fuse the results using softmax
		for var in self.vars_to_fuse:
			self.vars_fuse[var] = \
				tf.reduce_sum(
					self.vars_align[var]*ws,
					[2]
				)

		return 

	# align images in different res
	def align_maps(self, vars_to_fuse = None):
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
							tf.expand_dims(
								self.vars[i][var],-1
							),0
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

	def align_maps_linear(self, vars_to_fuse = None):
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
							tf.expand_dims(
								self.vars[i][var],-1
							),0
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

	def align_maps_cubic(self, vars_to_fuse = None):
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
					tf.image.resize_bicubic(\
						tf.expand_dims(
							tf.expand_dims(
								self.vars[i][var],-1
							),0
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

	# cutting out invalid areas
	def valid_windowed_region(self):
		# find out valid regions for the pyramids
		vars_to_cut = [\
			'u_1w','u_2w','u_3w','u_4w','Zw','confw'\
		]

		for i in range(len(self.cfg)):
			if PADDING == 'SAME':
				if self.cfg[0]['separable']:
					rows_cut = int(
						(self.cfg[i]['wx'].shape[1]-1)/2
					)
					cols_cut = int(
						(self.cfg[i]['wy'].shape[0]-1)/2
					)
				else:
					rows_cut = int(
						(self.cfg[i]['w'].shape[1]-1)/2
					)
					cols_cut = int(
						(self.cfg[i]['w'].shape[0]-1)/2
					)
				rows = self.cfg[i]['szx_sensor']
				cols = self.cfg[i]['szy_sensor']
			elif PADDING == 'VALID':
				rows_cut = 0
				cols_cut = 0
				rows = self.cfg[i]['szx_sensor']\
					 - (self.cfg[i]['wx'].shape[1]-1)
				cols = self.cfg[i]['szy_sensor']\
					 - (self.cfg[i]['wy'].shape[0]-1)
			
			for var in vars_to_cut:
				self.vars[i][var+'_valid'] = \
					self.vars[i][var][
						cols_cut:cols-cols_cut,
						rows_cut:rows-rows_cut
					]
				self.vars[i][var+'_valid_flat'] = \
					tf.reshape(
						self.vars[i][var+'_valid'], [-1]
					)

		# cut out the bad parts
		vars_to_cut = [\
			'u_1','u_2','u_3','u_4','Z','conf',\
		]

		for i in range(len(self.cfg)):
			rows_cut = int((self.cfg[i]['gauss'].shape[0]-1)/2)
			cols_cut = int((self.cfg[i]['gauss'].shape[1]-1)/2)
			rows = self.cfg[i]['szx_sensor']
			cols = self.cfg[i]['szy_sensor']
			
			for var in vars_to_cut:
				self.vars[i][var] = \
					self.vars[i][var][
						cols_cut:cols-cols_cut,
						rows_cut:rows-rows_cut
					]

		return 

	# cutting out invalid areas for fused data
	def valid_windowed_region_fuse(self):
		vars_to_cut = [\
			'Zw','confw'\
		]

		if PADDING == 'SAME':
			if self.fused_cfg['separable']:
				rows_cut = int(
					(self.fused_cfg['wx'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.fused_cfg['wy'].shape[0]-1)/2
				)
			else:
				rows_cut = int(
					(self.fused_cfg['w'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.fused_cfg['w'].shape[0]-1)/2
				)
			rows = self.cfg[0]['szx_sensor']
			cols = self.cfg[0]['szy_sensor']
		elif PADDING == 'VALID':
			rows_cut = 0
			cols_cut = 0
			rows = self.cfg[0]['szx_sensor']\
				 - (self.fused_cfg['wx'].shape[1]-1)
			cols = self.cfg[0]['szy_sensor']\
				 - (self.fused_cfg['wy'].shape[0]-1)
		
		for var in vars_to_cut:
			self.vars_fuse[var+'_valid'] = \
				self.vars_fuse[var][
					cols_cut:cols-cols_cut,
					rows_cut:rows-rows_cut
				]

			self.vars_fuse[var+'_valid_flat'] = \
				tf.reshape(
					self.vars_fuse[var+'_valid'], [-1]
				)

		# Z_gtw should be cut alone, since it does not have conv
		rows_cut = int(
			(self.fused_cfg['wx'].shape[1]-1)/2
		)
		cols_cut = int(
			(self.fused_cfg['wy'].shape[0]-1)/2
		)
		rows = self.cfg[0]['szx_sensor']
		cols = self.cfg[0]['szy_sensor']
		self.vars_fuse['Z_gtw_valid'] = \
			self.vars_fuse['Z_gt'][
				cols_cut:cols-cols_cut,
				rows_cut:rows-rows_cut
			]

		self.vars_fuse['Z_gtw_valid_flat'] = \
			tf.reshape(
				self.vars_fuse['Z_gtw_valid'], [-1]
			)

		# cut out the bad part
		vars_to_cut = [\
			'Z','conf','Z_gt'\
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

		return 

	# error functions
	def compute_err_all(self):
		# we will use the windowed version
		# WARNING: only suitable for planes 
		Zw_err_flat = []
		for i in range(len(self.cfg)):
			Zw_err_flat.append(tf.reduce_mean( 
				tf.abs(
					self.vars[i]['Zw'] - self.vars_fuse['Z_gt'][0,0]
				)
			))
		self.vars_align['Zw_err_flat'] = tf.pack(
			Zw_err_flat, 0
		)
		
		return

	def two_norm_err_all(self):
		# Compute the half norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err_all()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.sqrt(tf.reduce_mean(self.vars_align['Zw_err_flat']**2))

	def one_norm_err_all(self):
		# Compute the half norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err_all()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.reduce_mean(self.vars_align['Zw_err_flat'])

	def half_norm_err_all(self):
		# Compute the half norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err_all()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.reduce_mean(tf.sqrt(self.vars_align['Zw_err_flat']+0.01))**2

	def ptone_norm_err_all(self):
		# Compute the half norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err_all()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.reduce_mean((self.vars_align['Zw_err_flat']+0.01)**0.1)**10

	def sig_err_all(self):
		# Compute the half norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err_all()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		delta = 0.05
		return tf.reduce_mean((2/(
			1+tf.exp(-tf.square(self.vars_align['Zw_err_flat']/delta))
			)-1))

	def compute_err(self):
		# compute the fused error
		self.vars_fuse['Z_err'] = tf.abs(
			self.vars_fuse['Z'] - self.vars_fuse['Z_gt']
		)
		self.vars_fuse['Z_err_flat'] = tf.reshape(
			self.vars_fuse['Z_err'],[-1]
		)
		return

	def half_norm_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.reduce_mean(tf.sqrt(self.vars_fuse['Z_err_flat']+0.01))**2

	def half_norm_err1(self):
		return self.half_norm_err()

	def half_norm_err2(self):
		return self.half_norm_err()

	def sig_err(self):
		# Compute the sigmoid of the error for each pixel
		self.compute_err()
		delta = 0.03
		return tf.reduce_mean((2/(
			1+tf.exp(-tf.square(self.vars_fuse['Z_err_flat']/delta))
			)-1))

	def sparsification_err(self):
		# this one computes error as the area of the sparsification graph
		# for unwindowed version only
		self.compute_err()
		Z_err_flat = self.vars_fuse['Z_err_flat']
		conf_flat = tf.reshape(
			self.vars_fuse['conf'],[-1]
		)

		rows_cut = 1
		cols_cut = 1
		for i in range(len(self.cfg)):
			rows_cut *= int(self.cfg[i]['gauss'].shape[0]-1)
			cols_cut *= int(self.cfg[i]['gauss'].shape[0]-1)

		k = (self.resolution[0][0] - rows_cut) * \
			(self.resolution[0][1] - cols_cut)

		conf_sorted = tf.nn.top_k(conf_flat, k=k, sorted=True)
		Z_err_flat_sorted = tf.gather(\
			Z_err_flat, \
			conf_sorted.indices
		)
		self.vars_fuse['Z_err_flat_sorted'] = Z_err_flat_sorted
		self.vars_fuse['conf_flat_sorted'] = conf_sorted.values
		weight = np.arange(k)+1
		weight = 1/weight
		for i in range(len(weight)-1,0,-1):
			weight[i-1] += weight[i]
		weight = tf.constant(weight, dtype=tf.float32)

		# for the 0.5 norm, adding 0.01 is for numerical stability
		# return weight * tf.sqrt(Z_err_flat_sorted+0.01) / k
		return tf.reduce_mean(weight * Z_err_flat_sorted)

	def sparsification_err1(self):
		return self.sparsification_err()

	def sparsification_err2(self):
		return self.sparsification_err()

	def sparsification_err3(self):
		return self.sparsification_err()	

	# 
	def compute_err_layer(self, i):
		# we will use the windowed version
		# WARNING: only suitable for planes 
		self.vars[i]['Zw_err_flat'] = tf.abs(
			self.vars[i]['Zw'] - self.vars_fuse['Z_gt'][0,0]
		)
		return

	def two_norm_err_layer(self, i):
		self.compute_err_layer(i)
		return tf.sqrt(tf.reduce_mean(self.vars[i]['Zw_err_flat']**2))

	def one_norm_err_layer(self, i):
		self.compute_err_layer(i)
		return tf.reduce_mean(self.vars[i]['Zw_err_flat'])

	def half_norm_err_layer(self, i):
		self.compute_err_layer(i)
		return tf.reduce_mean(self.vars[i]['Zw_err_flat']**0.5)**2

	def ptone_norm_err_layer(self, i):
		self.compute_err_layer(i)
		return tf.reduce_mean(self.vars[i]['Zw_err_flat']**0.1)**10

	def two_norm_err_layer0(self):
		return self.two_norm_err_layer(0)

	def two_norm_err_layer1(self):
		return self.two_norm_err_layer(1)

	def two_norm_err_layer2(self):
		return self.two_norm_err_layer(2)

	def two_norm_err_layer3(self):
		return self.two_norm_err_layer(3)

	def two_norm_err_layer4(self):
		return self.two_norm_err_layer(4)

	def one_norm_err_layer0(self):
		return self.one_norm_err_layer(0)

	def one_norm_err_layer1(self):
		return self.one_norm_err_layer(1)

	def one_norm_err_layer2(self):
		return self.one_norm_err_layer(2)

	def one_norm_err_layer3(self):
		return self.one_norm_err_layer(3)

	def one_norm_err_layer4(self):
		return self.one_norm_err_layer(4)

	def half_norm_err_layer0(self):
		return self.half_norm_err_layer(0)

	def half_norm_err_layer1(self):
		return self.half_norm_err_layer(1)

	def half_norm_err_layer2(self):
		return self.half_norm_err_layer(2)

	def half_norm_err_layer3(self):
		return self.half_norm_err_layer(3)

	def half_norm_err_layer4(self):
		return self.half_norm_err_layer(4)

	def compute_angle(self, i):
		# This function computes the angle from each point (Z_gt, Z_est) 
		# to the (mu_f,mu_f) point
		stable_factor = 1e-5
		x_dir = self.cfg[i]['a1'] + self.cfg[i]['Z_0'] - self.vars_fuse['Z_gt'][0,0] + stable_factor
		y_dir = self.cfg[i]['a1'] + self.cfg[i]['Z_0'] - self.vars[i]['Zw'] + stable_factor
		self.vars[i]['Z_err_valid'] = tf.atan(x_dir/y_dir)*180.0/np.pi-45
		self.vars[i]['Z_err_valid_flat'] = tf.reshape(
			self.vars[i]['Z_err_valid'],[-1]
		)
		return

	def tan_square_err(self, i):
		self.compute_angle(i)
		return tf.square(
			tf.tan(self.vars[i]['Z_err_valid_flat'] / 180 * np.pi)
		)

	def tan_square_err0(self):
		return self.tan_square_err(0)

	def tan_square_err1(self):
		return self.tan_square_err(1)
		
	def tan_square_err2(self):
		return self.tan_square_err(2)

	def tan_square_err3(self):
		return self.tan_square_err(3)

	def tan_square_err4(self):
		return self.tan_square_err(4)	

	# query results
	def regular_output(self, log = False):
		res_dict = {}
		for k in self.image_to_show:
			res_dict[k] = self.vars_fuse[k]

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
			res_dict[k] = self.vars_fuse[k]
		self.results = self.session.run(res_dict)
		return self.results

	def query_results_layered(self, query_list):
		# query results from layered 
		res_dict = [{} for i in range(len(self.cfg))]
		self.results_layered = []
		for i in range(len(self.cfg)):
			for k in query_list:
				res_dict[i][k] = self.vars[i][k]
			self.results_layered.append(
				self.session.run(res_dict[i])
			)
		return self.results_layered

	def error_conf_map(self, Z_flat, Z_gt_flat, conf_flat, fig, s1,s2,s3, fig_name):
		# this functions draws the error and confidence map
		# compute average error
		err = np.abs(Z_flat - Z_gt_flat)

		# draw a fig that shows the average error with confidence > a threshold
		# cut the confidence region into bin_nums bins, with starting and ending
		# point in the center of the first and last bin respectively
		bin_nums = 100
		step = (
			conf_flat.max().astype(np.float64)-
			conf_flat.min().astype(np.float64)
			)/(bin_nums-1)
		conf_ranges = [conf_flat.min()-step/2, conf_flat.max()+step/2]

		cedgs = np.linspace(conf_flat.min(), conf_flat.max(), 100, True)
		err_sum = [0 for i in range(bin_nums)]
		err_count = [0 for i in range(bin_nums)]

		for i in range(bin_nums):
			lo = conf_ranges[0] + step * i
			hi = conf_ranges[0] + step * (i+1)
			flg1 = conf_flat.astype(np.float64) >= lo
			flg2 = conf_flat.astype(np.float64) < hi
			idx = np.where(flg1 * flg2)
			err_sum[i] = np.sum(err[idx])
			err_count[i] = len(idx[0])

		for i in range(bin_nums-1, 0, -1):
			err_sum[i-1] += err_sum[i]
			err_count[i-1] += err_count[i]

		for i in range(bin_nums):
			if err_count[i] != 0:
				err_sum[i] /= err_count[i]
				err_count[i] /= len(unconf_flat)

		# draw the figure
		title = fig_name
		ax1 = fig.add_subplot(s1,s2,s3, title=title)
		ax1.plot(cedgs, err_sum, 'b-')
		ax1.set_xlabel('Confidence threshold x')
		ax1.set_ylabel('Average error of conf > x (m)', color='b')

		ax2 = ax1.twinx()
		ax2.plot(cedgs, err_count, 'r-')
		ax2.set_ylabel('Ratio of conf > x', color='r')

		return 

	def error_conf_map_log(self, Z_flat, Z_gt_flat, conf_flat, fig, s1,s2,s3, fig_name):
		# this functions draws the error and confidence map using log scale of unconf
		# ie max(conf)+delta-conf
		unconf_flat = conf_flat.max() + 1e-5 - conf_flat
		# compute average error
		err = np.abs(Z_flat - Z_gt_flat)

		# draw a fig that shows the average error with confidence > a threshold
		# cut the confidence region into bin_nums bins, with starting and ending
		# point in the center of the first and last bin respectively
		bin_nums = 100
		step = (
			unconf_flat.max().astype(np.float64)/
			unconf_flat.min().astype(np.float64)
			)**(1/(bin_nums-1))
		conf_ranges = [unconf_flat.min()/np.sqrt(step), unconf_flat.max()*np.sqrt(step)]

		cedgs = np.logspace(0., 99., num=100, endpoint=True, base=step)*unconf_flat.min()
		err_sum = [0 for i in range(bin_nums)]
		err_count = [0 for i in range(bin_nums)]		

		for i in range(bin_nums):
			lo = conf_ranges[0] * step**i
			hi = conf_ranges[0] * step**(i+1)
			flg1 = unconf_flat.astype(np.float64) >= lo
			flg2 = unconf_flat.astype(np.float64) < hi
			idx = np.where(flg1 * flg2)
			err_sum[i] = np.sum(err[idx])
			err_count[i] = len(idx[0])

		for i in range(bin_nums-1):
			err_sum[i+1] += err_sum[i]
			err_count[i+1] += err_count[i]

		for i in range(bin_nums):
			if err_count[i] != 0:
				err_sum[i] /= err_count[i]
				err_count[i] /= len(unconf_flat)

		# draw the figure
		x_tmp = np.arange(len(err_sum))
		cedgs = conf_flat.max() + 1e-5 - cedgs
		cedgs = np.flipud(cedgs)
		cedgs = ["{:.4f}".format(i) for i in cedgs]
		err_sum.reverse()
		err_count.reverse()

		title = fig_name+', max conf: '+str(conf_flat.max())
		ax1 = fig.add_subplot(s1,s2,s3, title=title)
		plt.xticks(x_tmp[0::11], cedgs[0::11])
		ax1.plot(x_tmp, err_sum, 'b-')
		ax1.set_xlabel('Unconfidence threshold x')
		ax1.set_ylabel('Average error of unconf < x (m)', color='b')

		ax2 = ax1.twinx()
		ax2.plot(x_tmp, err_count, 'r-')
		ax2.set_ylabel('Ratio of unconf < x', color='r')

		return 

	def sparsification_plt(self, Z_flat, Z_gt_flat, conf_flat, fig, s1,s2,s3, fig_name):
		# this functions draws the error and confidence map
		# compute average error
		err = np.abs(Z_flat - Z_gt_flat).astype(np.float64)

		# sort the conf_flat
		err_sorted = err[np.argsort(conf_flat)]
		sparse = np.arange(len(err))/len(err)
		num = len(err) - np.arange(len(err))

		for i in range(len(err)-1,0,-1):
			err_sorted[i-1] += err_sorted[i]
		err_sorted /= num

		# draw a fig that shows the average error with a certain sparsication
		bin_nums = 1000
		step = np.linspace(0, len(err_sorted)-1, bin_nums, True).astype(np.int)
		
		err_show = err_sorted[step]
		sparse_show = sparse[step]

		# compute the AUC
		area = np.mean(err_sorted)

		# draw the figure
		title = fig_name
		ax1 = fig.add_subplot(s1,s2,s3, title=title+", area: "+str(area))
		ax1.plot(sparse_show, err_show, '-')
		ax1.set_xlabel('Sparsification')
		ax1.set_ylabel('Average error')

		return 

	def area_under_spars_curve(self, Z_flat, Z_gt_flat, conf_flat):
		# this functions draws the error and confidence map
		# compute average error
		err = np.abs(Z_flat - Z_gt_flat).astype(np.float64)

		# sort the conf_flat
		err_sorted = err[np.argsort(conf_flat)]
		sparse = np.arange(len(err))/len(err)
		num = len(err) - np.arange(len(err))

		for i in range(len(err)-1,0,-1):
			err_sorted[i-1] += err_sorted[i]
		err_sorted /= num

		# 
		return np.mean(err_sorted)

	"""training"""
	def update_loss(self):
		tmp_loss = np.sum(\
			self.session.run(
				self.vars_fuse['loss_func']\
				[self.cur_err_func]
			)
		)
		self.loss += tmp_loss
		return

	def update_der(self):
		for i in range(len(self.cfg)):
			tmp_der = self.session.run(
				self.output_grads[i]\
				[self.cur_err_func]
			)
			# append the derivative to the list
			self.list_der[i].append(tmp_der)
			# sum over the derivatives
			for key in tmp_der.keys():
				if key not in self.ave_der[i]:
					self.ave_der[i][key] = tmp_der[key]
				else:
					self.ave_der[i][key] += tmp_der[key]
		return 

	def finalize_der(self):
		# this function finalize the derivative and inputs 
		# it into the graph
		self.der_f_out = []
		for i in range(len(self.cfg)):
			self.der_f_out.append({})
			num_inst = len(self.list_der[i])
			# input the average derivative
			for key in self.ave_der[i].keys():
				self.der_f_out[i][self.input_der[i][key]] = \
					self.ave_der[i][key]/num_inst
			# 
			self.session.run(\
				self.assign_der_f[i][self.cur_err_func],\
				self.der_f_out[i]\
			)
		return

	def clear_der(self):
		# clear the past result
		self.list_der = \
			[[] for i in range(len(self.cfg))]
		self.ave_der = \
			[{} for i in range(len(self.cfg))]
		self.loss = 0
		return

	def get_old_var(self):
		self.old_var = []
		for i in range(len(self.cfg)):
			self.old_var.append(\
				self.session.run(
					self.output_vars[i][self.cur_err_func]
				)
			)
		return

	def update_apply_var(self, step):
		# this function computes the new values of variables and
		# input it into the graph
		self.new_var = []
		for i in range(len(self.cfg)):
			self.new_var.append({})
			for key in self.old_var[i].keys():
				self.new_var[i][self.input_var[i][key]] = \
					self.old_var[i][key] - \
					self.der_f_out[i][self.input_der[i][key]]*step
				self.session.run(\
					self.assign_new_var[i][self.cur_err_func], \
					self.new_var[i]\
				)
		return

	def print_loss(self):
		num_inst = len(self.list_der[0])
		tmp = self.loss/num_inst
		# clear the loss
		self.loss = 0
		print("Current average loss: ", tmp)
		return tmp

	def print_grads_and_vars(self):
		temp = []
		for i in range(len(self.cfg)):
			temp.append(self.session.run(\
				self.grads_and_vars[i][self.cur_err_func]\
			))
			print("Current grads and vars for",i,"th pyramid:",\
				temp[i]\
			)
		return temp

	def one_round_loss(self, I, Loc):
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# Update the loss
			self.update_loss()
		return self.print_loss()

	def one_step_training_force(self, I, Loc, step, min_step, temperature = 0.0):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss

		# self.visual_heatmap_percent(I, I_lap, Loc, per_thre=0.99)
		# self.visual_err_conf_map(I, I_lap, Loc, log=False)
		# self.sparsification_map(I, I_lap, Loc)
		# self.AUC_map(I, I_lap, Loc)
		# pdb.set_trace()

		print("Below is a new optimization step")

		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# # Initialization of recording for faster visualization
			# query_list = ['conf','Z','dZ_fusedws','dZ_fusedZ']
			# res = self.query_results(query_list)

			# query_list_layered = [\
			# 	'dZda0', 'dZ_fuseda0','dLda0','Z_align','dZ_fusedZ','conf_align','ws_align',\
			# 	'dconfda0', 'dZ_fusedconf', 'dwsdconf', 'dwsda0'
			# ]
			# res_layered = self.query_results_layered(query_list_layered)
			# for i in range(len(res_layered)):
			# 	if math.isnan(res_layered[i]['dLda0']):
			# 		pdb.set_trace()

			# Update the derivative
			self.update_loss()
			self.update_der()

		print("Parameters before moving:")
		old_loss = self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# self.visual_heatmap(I, Loc, conf_thre=-np.inf)

		# starting to find a good step size
		# we add some simulated annealing to the function
		print("Start to find a good step size")
		self.get_old_var()
		while(step > min_step):
			print("Current step size:", step)
			self.update_apply_var(step)
			# run one round
			new_loss = self.one_round_loss(I, Loc)
			thre = np.exp(
				((old_loss-new_loss)/old_loss-1e-6)/
				(temperature)
			)
			if new_loss <= old_loss:
				print("Accept lower loss")
				step = step * 2
				break
			elif np.random.rand() < thre:
				print("Accept random step, threshold: ", thre)
				step = step / 2
				break
			step = step / 2
			
		# if the step grows too small
		# turn back to the old variables
		if step <= min_step:
			step = 0
			self.update_apply_var(step)

		self.clear_der()
		return step, old_loss

	def visual_heatmap(self, I, Loc, conf_thre=0.):
		# since the confidence is not necessary max to 1
		# conf_thre is the threshold of the max confidence
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)

		num_w = len(res['Zw_valid'].flatten())
		idx_w = 0
		Zw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		confw_flat = np.empty((num_w*len(I),), dtype = np.float32)


		query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
		res_layered = self.query_results_layered(query_list_layered)

		num_unw_l = [len(res_layered[j]['Z'].flatten()) for j in range(len(self.cfg))]
		idx_unw_l = [0 for j in range(len(self.cfg))]
		Z_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gt_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		conf_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		num_w_l = [len(res_layered[j]['Zw_valid'].flatten()) for j in range(len(self.cfg))]
		idx_w_l = [0 for j in range(len(self.cfg))]
		Zw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gtw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		confw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			Zw_flat[idx_w:idx_w+num_w] = res['Zw_valid'].flatten()
			Z_gtw_flat[idx_w:idx_w+num_w] = res['Z_gtw_valid'].flatten()
			confw_flat[idx_w:idx_w+num_w] = res['confw_valid'].flatten()

			idx_unw += num_unw
			idx_w += num_w

			query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					np.ones(\
						Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]].shape
					) * Z_gt
				conf_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['conf'].flatten()

				Zw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['Zw_valid'].flatten()
				Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					np.ones(\
						Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]].shape
					) * Z_gt
				confw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['confw_valid'].flatten()

				idx_unw_l[j] += num_unw_l[j]
				idx_w_l[j] += num_w_l[j]

		# throw away all points with confidence smaller than conf_thre
		idx = np.where(conf_flat > conf_thre)
		Z_flat = Z_flat[idx]
		Z_gt_flat = Z_gt_flat[idx]
		conf_flat = conf_flat[idx]

		idx = np.where(confw_flat > conf_thre)
		Zw_flat = Zw_flat[idx]
		Z_gtw_flat = Z_gtw_flat[idx]
		confw_flat = confw_flat[idx]

		for j in range(len(self.cfg)):
			idx = np.where(conf_flat_layered[j] > conf_thre)
			Z_flat_layered[j] = Z_flat_layered[j][idx]
			Z_gt_flat_layered[j] = Z_gt_flat_layered[j][idx]
			conf_flat_layered[j] = conf_flat_layered[j][idx]

			idx = np.where(confw_flat_layered[j] > conf_thre)
			Zw_flat_layered[j] = Zw_flat_layered[j][idx]
			Z_gtw_flat_layered[j] = Z_gtw_flat_layered[j][idx]
			confw_flat_layered[j] = confw_flat_layered[j][idx]

		# draw the histograms
		fig = plt.figure()
		self.heatmap(\
			Z_flat, 
			Z_gt_flat, 
			fig, 
			3,4,1, 
			'fused_result'
		)
		self.heatmap(\
			Zw_flat, 
			Z_gtw_flat, 
			fig, 
			3,4,7, 
			'fused_result, window '+str(self.fused_cfg['w'].shape)
		)
		for i in range(len(self.cfg)):
			self.heatmap(\
				Z_flat_layered[i], 
				Z_gt_flat_layered[i], 
				fig, 
				3,4,i+2, 
				'layer '+str(i)
			)
			self.heatmap(\
				Zw_flat_layered[i], 
				Z_gtw_flat_layered[i], 
				fig, 
				3,4,i+8, 
				'layer '+str(i)+', window '+str(self.cfg[i]['w'].shape)
			)

		plt.show()
		return 

	def visual_heatmap_percent(self, I, Loc, per_thre=0.):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		Z_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		conf_flat = np.empty((num_unw,len(I)), dtype = np.float32)

		num_w = len(res['Zw_valid'].flatten())
		Zw_flat = np.empty((num_w,len(I)), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w,len(I)), dtype = np.float32)
		confw_flat = np.empty((num_w,len(I)), dtype = np.float32)


		query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
		res_layered = self.query_results_layered(query_list_layered)

		num_unw_l = [len(res_layered[j]['Z'].flatten()) for j in range(len(self.cfg))]
		Z_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gt_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		conf_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		num_w_l = [len(res_layered[j]['Zw_valid'].flatten()) for j in range(len(self.cfg))]
		Zw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gtw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		confw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
			res = self.query_results(query_list)

			Z_flat[:,i] = res['Z'].flatten()
			Z_gt_flat[:,i] = res['Z_gt'].flatten()
			conf_flat[:,i] = res['conf'].flatten()

			Zw_flat[:,i] = res['Zw_valid'].flatten()
			Z_gtw_flat[:,i] = res['Z_gtw_valid'].flatten()
			confw_flat[:,i] = res['confw_valid'].flatten()

			query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][:,i] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][:,i] = \
					np.ones(Z_gt_flat_layered[j][:,i].shape) * Z_gt
				conf_flat_layered[j][:,i] = \
					res_layered[j]['conf'].flatten()

				Zw_flat_layered[j][:,i] = \
					res_layered[j]['Zw_valid'].flatten()
				Z_gtw_flat_layered[j][:,i] = \
					np.ones(Z_gtw_flat_layered[j][:,i].shape) * Z_gt
				confw_flat_layered[j][:,i] = \
					res_layered[j]['confw_valid'].flatten()

		# throw away all points with confidence smaller than a certain percent
		idx0 = np.argsort(conf_flat, axis=0)
		idx1 = np.ones((conf_flat.shape[0],1),dtype=np.int) * \
			np.array([np.arange(len(I))])
		idx = (idx0.flatten(), idx1.flatten().astype(np.int))
		cut_row = int(Z_flat.shape[0] * per_thre)

		Z_flat = np.reshape(Z_flat[idx], Z_flat.shape)
		Z_gt_flat = np.reshape(Z_gt_flat[idx], Z_gt_flat.shape)
		Z_flat = Z_flat[cut_row:,:].flatten()
		Z_gt_flat = Z_gt_flat[cut_row:,:].flatten()


		idx0 = np.argsort(confw_flat, axis=0)
		idx1 = np.ones((confw_flat.shape[0],1),dtype=np.int) * \
			np.array([np.arange(len(I))])
		idx = (idx0.flatten(), idx1.flatten())
		cut_row = int(Zw_flat.shape[0] * per_thre)

		Zw_flat = np.reshape(Zw_flat[idx], Zw_flat.shape)
		Z_gtw_flat = np.reshape(Z_gtw_flat[idx], Z_gtw_flat.shape)
		Zw_flat = Zw_flat[cut_row:,:].flatten()
		Z_gtw_flat = Z_gtw_flat[cut_row:,:].flatten()


		for j in range(len(self.cfg)):
			idx0 = np.argsort(conf_flat_layered[j], axis=0)
			idx1 = np.ones((conf_flat_layered[j].shape[0],1),dtype=np.int) * \
				np.array([np.arange(len(I))])
			idx = (idx0.flatten(), idx1.flatten())
			cut_row = int(Z_flat_layered[j].shape[0] * per_thre)

			Z_flat_layered[j] = np.reshape(\
				Z_flat_layered[j][idx],\
				Z_flat_layered[j].shape
			)
			Z_gt_flat_layered[j] = np.reshape(\
				Z_gt_flat_layered[j][idx],\
				Z_gt_flat_layered[j].shape
			)
			Z_flat_layered[j] = Z_flat_layered[j][cut_row:,:].flatten()
			Z_gt_flat_layered[j] = Z_gt_flat_layered[j][cut_row:,:].flatten()


			idx0 = np.argsort(confw_flat_layered[j], axis=0)
			idx1 = np.ones((confw_flat_layered[j].shape[0],1),dtype=np.int) * \
				np.array([np.arange(len(I))])
			idx = (idx0.flatten(), idx1.flatten())
			cut_row = int(Zw_flat_layered[j].shape[0] * per_thre)

			Zw_flat_layered[j] = np.reshape(\
				Zw_flat_layered[j][idx],\
				Zw_flat_layered[j].shape
			)
			Z_gtw_flat_layered[j] = np.reshape(\
				Z_gtw_flat_layered[j][idx],\
				Z_gtw_flat_layered[j].shape
			)
			Zw_flat_layered[j] = Zw_flat_layered[j][cut_row:,:].flatten()
			Z_gtw_flat_layered[j] = Z_gtw_flat_layered[j][cut_row:,:].flatten()


		# draw the histograms
		fig = plt.figure()
		self.heatmap(\
			Z_flat, 
			Z_gt_flat, 
			fig, 
			3,4,1, 
			'fused_result'
		)
		self.heatmap(\
			Zw_flat, 
			Z_gtw_flat, 
			fig, 
			3,4,7, 
			'fused_result, window '+str(self.fused_cfg['w'].shape)
		)
		for i in range(len(self.cfg)):
			self.heatmap(\
				Z_flat_layered[i], 
				Z_gt_flat_layered[i], 
				fig, 
				3,4,i+2, 
				'layer '+str(i)
			)
			self.heatmap(\
				Zw_flat_layered[i], 
				Z_gtw_flat_layered[i], 
				fig, 
				3,4,i+8, 
				'layer '+str(i)+', window '+str(self.cfg[i]['w'].shape)
			)

		plt.show()
		return 

	def visual_err_conf_map(self, I, Loc, log=True):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)


		# Initialization for recording for faster visualization
		query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)

		num_w = len(res['Zw_valid'].flatten())
		idx_w = 0
		Zw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		confw_flat = np.empty((num_w*len(I),), dtype = np.float32)


		query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
		res_layered = self.query_results_layered(query_list_layered)

		num_unw_l = [len(res_layered[j]['Z'].flatten()) for j in range(len(self.cfg))]
		idx_unw_l = [0 for j in range(len(self.cfg))]
		Z_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gt_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		conf_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		num_w_l = [len(res_layered[j]['Zw_valid'].flatten()) for j in range(len(self.cfg))]
		idx_w_l = [0 for j in range(len(self.cfg))]
		Zw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gtw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		confw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			Zw_flat[idx_w:idx_w+num_w] = res['Zw_valid'].flatten()
			Z_gtw_flat[idx_w:idx_w+num_w] = res['Z_gtw_valid'].flatten()
			confw_flat[idx_w:idx_w+num_w] = res['confw_valid'].flatten()

			idx_unw += num_unw
			idx_w += num_w

			query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					np.ones(
						Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]].shape
					)* Z_gt
				conf_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['conf'].flatten()

				Zw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['Zw_valid'].flatten()
				Z_map_gtw = np.ones(res_layered[j]['Zw_valid'].shape) * Z_gt
				Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					np.ones(
						Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]].shape
					)* Z_gt
				confw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['confw_valid'].flatten()

				idx_unw_l[j] += num_unw_l[j]
				idx_w_l[j] += num_w_l[j]

		# draw the performance measurement
		if log:
			fig = plt.figure()
			self.error_conf_map_log(\
				Z_flat, 
				Z_gt_flat, 
				conf_flat, 
				fig, 
				3,3,1, 
				'fused result'
			)
			self.error_conf_map_log(\
				Zw_flat, 
				Z_gtw_flat, 
				confw_flat, 
				fig, 
				3,3,7, 
				'fused_result, window '+str(self.fused_cfg['w'].shape)
			)
			for i in range(len(self.cfg)):
				self.error_conf_map_log(\
					Z_flat_layered[i],
					Z_gt_flat_layered[i],
					conf_flat_layered[i],
					fig,
					3,3,i+2,
					'layer '+str(i)
				)
				# self.error_conf_map_log(\
				# 	Zw_flat_layered[i],
				# 	Z_gtw_flat_layered[i],
				# 	confw_flat_layered[i],
				# 	fig,
				# 	3,4,i+8,
				# 	'layer '+str(i)+', window '+str(self.cfg[i]['w'].shape)
				# )
		else:
			fig = plt.figure()
			self.error_conf_map(\
				Z_flat, 
				Z_gt_flat, 
				conf_flat, 
				fig, 
				3,3,1, 
				'fused result'
			)
			self.error_conf_map(\
				Zw_flat, 
				Z_gtw_flat, 
				confw_flat, 
				fig, 
				3,3,7, 
				'fused_result, window '+str(self.fused_cfg['w'].shape)
			)
			for i in range(len(self.cfg)):
				self.error_conf_map(\
					Z_flat_layered[i],
					Z_gt_flat_layered[i],
					conf_flat_layered[i],
					fig,
					3,3,i+2,
					'layer '+str(i)
				)
				# self.error_conf_map(\
				# 	Zw_flat_layered[i],
				# 	Z_gtw_flat_layered[i],
				# 	confw_flat_layered[i],
				# 	fig,
				# 	3,4,i+8,
				# 	'layer '+str(i)+', window '+str(self.cfg[i]['w'].shape)
				# )

		plt.show()
		return 

	def sparsification_map(self, I, Loc):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)


		# Initialization for recording for faster visualization
		query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)

		num_w = len(res['Zw_valid'].flatten())
		idx_w = 0
		Zw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w*len(I),), dtype = np.float32)
		confw_flat = np.empty((num_w*len(I),), dtype = np.float32)


		query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
		res_layered = self.query_results_layered(query_list_layered)

		num_unw_l = [len(res_layered[j]['Z'].flatten()) for j in range(len(self.cfg))]
		idx_unw_l = [0 for j in range(len(self.cfg))]
		Z_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gt_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		conf_flat_layered = [
			np.empty((num_unw_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		num_w_l = [len(res_layered[j]['Zw_valid'].flatten()) for j in range(len(self.cfg))]
		idx_w_l = [0 for j in range(len(self.cfg))]
		Zw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gtw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		confw_flat_layered = [
			np.empty((num_w_l[j]*len(I),), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			Zw_flat[idx_w:idx_w+num_w] = res['Zw_valid'].flatten()
			Z_gtw_flat[idx_w:idx_w+num_w] = res['Z_gtw_valid'].flatten()
			confw_flat[idx_w:idx_w+num_w] = res['confw_valid'].flatten()

			idx_unw += num_unw
			idx_w += num_w

			query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					np.ones(
						Z_gt_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]].shape
					) * Z_gt
				conf_flat_layered[j][idx_unw_l[j]:idx_unw_l[j]+num_unw_l[j]] = \
					res_layered[j]['conf'].flatten()

				Zw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['Zw_valid'].flatten()
				Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					np.ones(
						Z_gtw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]].shape
					) * Z_gt
				confw_flat_layered[j][idx_w_l[j]:idx_w_l[j]+num_w_l[j]] = \
					res_layered[j]['confw_valid'].flatten()

				idx_unw_l[j] += num_unw_l[j]
				idx_w_l[j] += num_w_l[j]

		# draw the performance measurement
		fig = plt.figure()
		self.sparsification_plt(\
			Z_flat, 
			Z_gt_flat, 
			conf_flat, 
			fig, 
			3,3,1, 
			'fused result'
		)
		self.sparsification_plt(\
			Zw_flat, 
			Z_gtw_flat, 
			confw_flat, 
			fig, 
			3,3,7, 
			'fused_result, window '+str(self.fused_cfg['w'].shape)
		)
		for i in range(len(self.cfg)):
			self.sparsification_plt(\
				Z_flat_layered[i],
				Z_gt_flat_layered[i],
				conf_flat_layered[i],
				fig,
				3,3,i+2,
				'layer '+str(i)
			)
			# self.error_conf_map_log(\
			# 	Zw_flat_layered[i],
			# 	Z_gtw_flat_layered[i],
			# 	confw_flat_layered[i],
			# 	fig,
			# 	3,4,i+8,
			# 	'layer '+str(i)+', window '+str(self.cfg[i]['w'].shape)
			# )

		plt.show()
		return 

	def AUC_map(self, I, Loc):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		Z_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		conf_flat = np.empty((num_unw,len(I)), dtype = np.float32)

		num_w = len(res['Zw_valid'].flatten())
		Zw_flat = np.empty((num_w,len(I)), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w,len(I)), dtype = np.float32)
		confw_flat = np.empty((num_w,len(I)), dtype = np.float32)


		query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
		res_layered = self.query_results_layered(query_list_layered)

		num_unw_l = [len(res_layered[j]['Z'].flatten()) for j in range(len(self.cfg))]
		Z_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gt_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		conf_flat_layered = [
			np.empty((num_unw_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		num_w_l = [len(res_layered[j]['Zw_valid'].flatten()) for j in range(len(self.cfg))]
		Zw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		Z_gtw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]
		confw_flat_layered = [
			np.empty((num_w_l[j],len(I)), dtype = np.float32)\
			for j in range(len(self.cfg))
		]

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf','Zw_valid','Z_gtw_valid', 'confw_valid']
			res = self.query_results(query_list)

			Z_flat[:,i] = res['Z'].flatten()
			Z_gt_flat[:,i] = res['Z_gt'].flatten()
			conf_flat[:,i] = res['conf'].flatten()

			Zw_flat[:,i] = res['Zw_valid'].flatten()
			Z_gtw_flat[:,i] = res['Z_gtw_valid'].flatten()
			confw_flat[:,i] = res['confw_valid'].flatten()

			query_list_layered = ['Z', 'Zw_valid', 'conf', 'confw_valid']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][:,i] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][:,i] = \
					np.ones(Z_gt_flat_layered[j][:,i].shape) * Z_gt
				conf_flat_layered[j][:,i] = \
					res_layered[j]['conf'].flatten()

				Zw_flat_layered[j][:,i] = \
					res_layered[j]['Zw_valid'].flatten()
				Z_gtw_flat_layered[j][:,i] = \
					np.ones(Z_gtw_flat_layered[j][:,i].shape) * Z_gt
				confw_flat_layered[j][:,i] = \
					res_layered[j]['confw_valid'].flatten()

		AUC = np.empty((Z_flat.shape[1],))
		AUCw = np.empty((Zw_flat.shape[1],))
		AUC_layered = [
			np.empty((Z_flat_layered[j].shape[1],))
			for j in range(len(self.cfg))
		]
		AUCw_layered = [
			np.empty((Zw_flat_layered[j].shape[1],))
			for j in range(len(self.cfg))
		]

		# compute the AUC
		for i in range(len(AUC)):
			# fig = plt.figure()
			# self.sparsification_plt(\
			# 	Z_flat[:,i], \
			# 	Z_gt_flat[:,i], \
			# 	conf_flat[:,i], \
			# 	fig, 1,1,1, \
			# 	"depth: "+str(Z_gt_flat[0,i])
			# )
			# plt.show()

			AUC[i] = self.area_under_spars_curve(
				Z_flat[:,i],
				Z_gt_flat[:,i],
				conf_flat[:,i]
			)

			AUCw[i] = self.area_under_spars_curve(
				Zw_flat[:,i],
				Z_gtw_flat[:,i],
				confw_flat[:,i]
			)

			for j in range(len(self.cfg)):
				AUC_layered[j][i] = self.area_under_spars_curve(
					Z_flat_layered[j][:,i],
					Z_gt_flat_layered[j][:,i],
					conf_flat_layered[j][:,i]
				)
				AUCw_layered[j][i] = self.area_under_spars_curve(
					Zw_flat_layered[j][:,i],
					Z_gtw_flat_layered[j][:,i],
					confw_flat_layered[j][:,i]
				)

		# plot the AUC
		fig = plt.figure()
		ax = fig.add_subplot(3,3,1, title="fused result")
		ax.plot(Z_gt_flat[0,:], AUC,'o')
		ax = fig.add_subplot(\
			3,3,7, \
			title="fused result, window: "+str(self.fused_cfg['w'].shape)
		)
		ax.plot(Z_gtw_flat[0,:], AUCw, 'o')
		for j in range(len(self.cfg)):
			ax = fig.add_subplot(3,3,j+2, title="layer "+str(j))
			ax.plot(Z_gt_flat_layered[j][0,:], AUC_layered[j], 'o')

		plt.show()

		
		# save the data
		lpickle = len(glob.glob('./test_results/pyConfLensFlowNetFast/*.pickle'))
		fileName = os.path.join(\
			'./test_results/pyConfLensFlowNetFast/'+str(lpickle)+".pickle"
		)
		with open(fileName,'wb') as f:
			cfg_data = {
				'Z_flat':				Z_flat,
				'Z_gt_flat':			Z_gt_flat,
				'conf_flat':			conf_flat,
				'Zw_flat':				Zw_flat,
				'Z_gtw_flat':			Z_gtw_flat,
				'confw_flat':			confw_flat,
				'Z_flat_layered':		Z_flat_layered,
				'Z_gt_flat_layered':	Z_gt_flat_layered,
				'conf_flat_layered':	conf_flat_layered,
				'Zw_flat_layered':		Zw_flat_layered,
				'Z_gtw_flat_layered':	Z_gtw_flat_layered,
				'confw_flat_layered':	confw_flat_layered,
				'AUC':					AUC,
				'AUCw':					AUCw,
				'AUC_layered':			AUC_layered,
				'AUCw_layered':			AUCw_layered,
			}
			# dump the data into the file
			pickle.dump(cfg_data, f)
