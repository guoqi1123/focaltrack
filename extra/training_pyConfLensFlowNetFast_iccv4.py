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
class training_pyConfLensFlowNetFast_iccv4(training_focalFlowNet):
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
			'a0a1' : 1, #sensor distance
			'da0a1_ratio': 1, #adjust the scale to converge faster
			'b0' : 1e-4, #
			'db1_ratio': 1, #adjust the scale to converge faster
			'b1' : -1.9, #
			'db1_ratio': 1, #adjust the scale to converge faster
			'Z_0': -0.2,
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
			'a0_o':'dLda0_o',
			'a1_o':'dLda1_o',
			'Z_0':'dLdZ_0',
			'ra0_1':'dLdra0_1',
			'ra0_2':'dLdra0_2',
			'ra1_1':'dLdra1_1',
			'ra1_2':'dLdra1_2',
			'rx_y':'dLdrx_y'
		}
		self.var_dict = {
			'dLdfave':'fave',
			'dLdft':'ft',
			'dLda0_o':'a0_o',
			'dLda1_o':'a1_o',
			'dLdZ_0': 'Z_0',
			'dLdra0_1':	'ra0_1',
			'dLdra0_2':	'ra0_2',
			'dLdra1_1':	'ra1_1',
			'dLdra1_2':	'ra1_2',
			'dLdrx_y':	'rx_y',
		}

		self.image_to_show = ['Z','conf','Z_gt']
		self.vars_to_fuse = ['Z','conf','Z_f','conff']

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
	def input_images(self, I, Z, offset):
		# import data into the network
		# I, I_lap_batch and Z should be a n-element tuple
		input_dict = {}
		input_dict[self.I_in] = I
		input_dict[self.Z_in] = Z
		input_dict[self.offset_in] = offset
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

			self.offset_in = tf.Variable(1, dtype=tf.float32)
			offset = tf.Variable(1, dtype=tf.float32)

			Z_gt0 = []

			Z_0 = []
			a0 = []
			a1 = []
			a0_o = []
			a1_o = []
			ra0_1 = []
			ra0_2 = []
			ra1_1 = []
			ra1_2 = []
			rx_y = []
			gauss = []
			ext_f = []
			ft = []
			fave = []
			if self.cfg[0]['separable']:
				wx = []
				wy = []
			else:
				w = []

			I_t = []
			I_lap = []

			u_1 = []
			u_2 = []
			u_3 = []
			u_4 = []
			u_3f = []
			u_4f = []
			Z = []
			Z_f = []
			self.a0 = []
			self.a1 = []

			self.Zs = []
			self.confs = []

			# depth computation
			# I used to be (960, 600, 2or 3) we change it to (2or3, 960, 600) 
			tmp_I = tf.transpose(I, perm=[2,0,1])
			for i in range(len(self.cfg)):
				# initialize variables				
				"""Input parameters"""
				Z_0.append(tf.Variable(self.cfg[i]['Z_0'], dtype = tf.float32))
				
				# compute a0
				a0_o.append(tf.Variable(self.cfg[i]['a0_o'], dtype=tf.float32))
				a1_o.append(tf.Variable(self.cfg[0]['a1_o'], dtype=tf.float32))

				o = tf.stack([(offset/10000)**k for k in range(len(self.cfg[i]['a0_o']))],0)
				a0.append(tf.reduce_sum(o*a0_o[i]))

				o = offset/10000
				a1.append(tf.exp(a1_o[i][0] * o + a1_o[i][1])+a1_o[i][2])
				# o = tf.stack([(offset/10000)**k for k in range(len(self.cfg[i]['a1_o']))],0)
				# a1.append(tf.reduce_sum(o*a1_o[i]))
				self.a0.append(a0[i])
				self.a1.append(a1[i])
				self.o = o
				self.offset = offset

				xx,yy = np.meshgrid(\
					np.arange(self.resolution[i][1]),
					np.arange(self.resolution[i][0])
				)

				# radial distortion
				xx = (xx - (self.resolution[i][1]-1)/2)/self.resolution[i][0]
				yy = (yy - (self.resolution[i][0]-1)/2)/self.resolution[i][0]
				rx_y.append(tf.Variable(self.cfg[0]['rx_y'], dtype=tf.float32))
				r = tf.sqrt(xx**2 + yy**2 * rx_y[0])
				ra0_1.append(tf.Variable(self.cfg[i]['ra0_1'], dtype=tf.float32))
				ra0_2.append(tf.Variable(self.cfg[i]['ra0_2'], dtype=tf.float32))
				ra1_1.append(tf.Variable(self.cfg[i]['ra1_1'], dtype=tf.float32))
				ra1_2.append(tf.Variable(self.cfg[i]['ra1_2'], dtype=tf.float32))

				a0_r = a0[i] + ra0_1[i] * r + ra0_2[i] * (r**2)
				a1_r = a1[0] + ra1_1[0] * r + ra1_2[0] * (r**2)

				gauss.append(tf.Variable(self.cfg[i]['gauss'], dtype = tf.float32))
				ext_f.append(tf.Variable(self.cfg[i]['ext_f'], dtype = tf.float32))
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

				# Generate the differential images
				I_t.append(dIdt(I_batch[i], ft[i]))
				I_lap.append(dIdt(I_lap_batch[i], fave[i]))


				# unwindowed version
				u_1.append(I_lap[i])
				u_2.append(I_t[i])
				u_3.append(a0_r * a1_r * u_1[i])
				u_4.append(-u_2[i] + a0_r * u_1[i])
				u_3f.append(a0_r * a1_r * u_1[i])
				u_4f.append(-u_2[i] + a0_r * u_1[i])

				ext_f[i] = tf.expand_dims(\
					tf.transpose(ext_f[i], perm=[1,2,0]),-2
				)
				u_3[i] = tf.expand_dims(tf.expand_dims(u_3[i],0),-1)
				u_4[i] = tf.expand_dims(tf.expand_dims(u_4[i],0),-1)
				u_3[i] = tf.nn.conv2d(u_3[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_4[i] = tf.nn.conv2d(u_4[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				Z.append((u_3[i]*u_4[i]) / (u_4[i]*u_4[i] + 1e-5) + Z_0[0])

				u_3f[i] = tf.expand_dims(tf.expand_dims(u_3f[i],0),-1)
				u_4f[i] = tf.expand_dims(tf.expand_dims(u_4f[i],0),-1)
				u_3f[i] = tf.nn.conv2d(u_3f[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				u_4f[i] = tf.nn.conv2d(u_4f[i], ext_f[i],[1,1,1,1],padding='SAME')[0,:,:,:]
				Z_f.append((u_3f[i]*u_4f[i]) / (u_4f[i]*u_4f[i] + 1e-5) + Z_0[0])
				
				#save references to required I/O]
				self.vars.append({})
				self.vars[i]['I'] = I_batch[i]
				
				# unwindowed version
				self.vars[i]['u_1'] = u_1[i]
				self.vars[i]['u_2'] = u_2[i]
				self.vars[i]['u_3'] = u_3[i]
				self.vars[i]['u_4'] = u_4[i]
				self.vars[i]['u_3f'] = u_3f[i]
				self.vars[i]['u_4f'] = u_4f[i]
				self.vars[i]['Z'] = Z[i]
				self.vars[i]['Z_f'] = Z_f[i]

			# align depth and confidence maps
			self.align_maps_ext(['u_3','u_4','u_3f','u_4f'])

			# compute the aligned version of Z
			self.vars_align['Z'] = \
				self.vars_align['u_3']*self.vars_align['u_4'] / \
				(self.vars_align['u_4']*self.vars_align['u_4'] + 1e-5) + Z_0[0]

			self.vars_align['Z_f'] = \
				self.vars_align['u_3f']*self.vars_align['u_4f'] / \
				(self.vars_align['u_4f']*self.vars_align['u_4f'] + 1e-5) + Z_0[0]

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[0]['conf_func']+'()')

			# save the ground truth
			self.vars_fuse['Z_gt'] = Z_gt
			Z_gt_tmp = [Z_gt for i in range(len(self.cfg))]
			self.vars_align['Z_gt'] = tf.stack(Z_gt_tmp, 2)

			# fusion
			self.softmax_fusion()	

			# cut out the region that do not work
			self.valid_windowed_region()
			self.valid_windowed_region_fuse()

			# record the fused Zs
			self.Zs.append(self.vars_fuse['Z'])
			self.Zs.append(self.vars_fuse['Z_f'])
			
			self.confs.append(self.vars_fuse['conf'])
			self.confs.append(self.vars_fuse['conff'])

			# compute error
			self.vars_fuse['loss_func'] = {}
			for ef in self.cfg[0]['err_func']:
				self.vars_fuse['loss_func'][ef] = \
					eval('self.'+ef+'()')	

			# obtain the final prediction
			vars_to_fuse = ['Z','conf']
			self.vars_fuse['Z'] = tf.reduce_sum(
				tf.stack(
					[\
						self.Zs[k]*self.ws[k]\
						for k in range(len(self.Zs))\
					],
				-1),
			-1)
			self.vars_fuse['conf'] = tf.reduce_sum(
				tf.stack(
					[\
						self.confs[k]*self.ws[k]\
						for k in range(len(self.confs))\
					],
				-1),
			-1)	

			# compute derivatives for prediction
			for i in range(len(self.cfg)):
				self.der.append({})
				self.der[i]['dLdft'] = {}
				self.vars[i]['ft'] = ft[i]			

				self.der[i]['dLdfave'] = {}
				self.vars[i]['fave'] = fave[i]

				self.der[i]['dLda0_o'] = {}
				self.der[i]['dLda1_o'] = {}
				self.der[i]['dLdZ_0'] = {}
				self.vars[i]['a0_o'] = a0_o[i]
				self.vars[i]['a1_o'] = a1_o[i]
				self.vars[i]['Z_0'] = Z_0[i]

				self.der[i]['dLdra0_1'] = {}
				self.der[i]['dLdra0_2'] = {}
				self.der[i]['dLdra1_1'] = {}
				self.der[i]['dLdra1_2'] = {}
				self.der[i]['dLdrx_y'] = {}
				self.vars[i]['ra0_1'] = ra0_1[i]
				self.vars[i]['ra0_2'] = ra0_2[i]
				self.vars[i]['ra1_1'] = ra1_1[i]
				self.vars[i]['ra1_2'] = ra1_2[i]
				self.vars[i]['rx_y'] = rx_y[i]

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
							dLdft, axis = [2]
						)
						dLdft = dLdft\
							+ tf.reverse(dLdft, axis = [0])\
						 	+ tf.reverse(dLdft, axis = [1])\
						 	+ tf.reverse(dLdft, axis = [0,1])
						dLdft = dLdft\
							+ tf.transpose(dLdft, perm = [1,0,2])
						dLdft = dLdft\
							+ tf.reverse(dLdft, axis = [0])
						dLdft = dLdft\
							+ tf.reverse(dLdft, axis = [1])
						dLdft = dLdft / 64.
						self.der[i]['dLdft'][ef] = dLdft
					
					if dLdfave == None:
						self.der[i]['dLdfave'][ef] = tf.zeros(tf.shape(self.vars[i]['fave']))
					else:
						# derivative w.r.t. to fave
						dLdfave = dLdfave
						self.der[i]['dLdfave'][ef] = dLdfave

					#### derivative w.r.t. optical parameters
					dLda0_o, dLda1_o, dLdZ_0 = tf.gradients(
						self.vars_fuse['loss_func'][ef],\
						[a0_o[i], a1_o[i], Z_0[i]]
					)
					if dLda0_o == None:
						dLda0_o = tf.constant(np.zeros(self.cfg[i]['a0_o'].shape), dtype=tf.float32)
					if dLda1_o == None:
						dLda1_o = tf.constant(np.zeros(self.cfg[i]['a1_o'].shape), dtype=tf.float32)
					if dLdZ_0 == None:
						dLdZ_0 = tf.constant(0,dtype =tf.float32)

					dLda0_o = dLda0_o * self.cfg[i]['da0_o_ratio']
					dLda1_o = dLda1_o * self.cfg[i]['da1_o_ratio']
					dLdZ_0= dLdZ_0 * self.cfg[i]['dZ_0_ratio']
					self.der[i]['dLda0_o'][ef] = dLda0_o
					self.der[i]['dLda1_o'][ef] = dLda1_o
					self.der[i]['dLdZ_0'][ef] = dLdZ_0

					#### derivative w.r.t. radial parameters
					dLdra0_1, dLdra0_2, dLdra1_1, dLdra1_2, dLdrx_y = tf.gradients(
						self.vars_fuse['loss_func'][ef],\
						[ra0_1[i], ra0_2[i], ra1_1[i], ra1_2[i], rx_y[i]]
					)
					if dLdra0_1 == None:
						dLdra0_1 = tf.constant(0,dtype =tf.float32)
					if dLdra0_2 == None:
						dLdra0_2 = tf.constant(0,dtype = tf.float32)
					if dLdra1_1 == None:
						dLdra1_1 = tf.constant(0,dtype =tf.float32)
					if dLdra1_2 == None:
						dLdra1_2 = tf.constant(0,dtype =tf.float32)
					if dLdrx_y == None:
						dLdrx_y = tf.constant(0,dtype =tf.float32)

					dLdra0_1 = dLdra0_1 * self.cfg[i]['dra0_1_ratio']
					dLdra0_2 = dLdra0_2 * self.cfg[i]['dra0_2_ratio']
					dLdra1_1 = dLdra1_1 * self.cfg[i]['dra1_1_ratio']
					dLdra1_2 = dLdra1_2 * self.cfg[i]['dra1_2_ratio']
					dLdrx_y = dLdrx_y * self.cfg[i]['drx_y_ratio']

					self.der[i]['dLdra0_1'][ef] = dLdra0_1
					self.der[i]['dLdra0_2'][ef] = dLdra0_2
					self.der[i]['dLdra1_1'][ef] = dLdra1_1
					self.der[i]['dLdra1_2'][ef] = dLdra1_2
					self.der[i]['dLdrx_y'][ef] = dLdrx_y

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

			# create an optimizer
			self.train = {}
			opt = tf.train.AdagradOptimizer(learning_rate=0.001)
			for ef in self.cfg[0]['err_func']:
				grads_and_vars = []
				for i in range(len(self.cfg)):
					grads_and_vars += self.grads_and_vars[i][ef]
				self.train[ef] = opt.apply_gradients(grads_and_vars)

			#add values
			#as number the inputs depends on num_py,
			#we use string operations to do it
			self.input_data = tf.group(\
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in),
				offset.assign(self.offset_in),
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

			# unwindowed version
			conf_tmp = []
			for j in range(self.cfg[0]['ext_f'].shape[0]):
				conf_tmp.append(
					(self.vars[i]['u_4'][:,:,j]**2 + 1e-20)/\
					tf.sqrt(\
						w_bc[i][j] * self.vars[i]['u_3'][:,:,j]**2 + \
						w_bc1[i][j] * self.vars[i]['u_4'][:,:,j]**2 + \
						w_bc2[i][j] * self.vars[i]['u_3'][:,:,j]*self.vars[i]['u_4'][:,:,j] + \
						self.vars[i]['u_4'][:,:,j]**4 + \
						1e-10\
					)
				)
			conf_tmp = tf.stack(conf_tmp, 2)

			# unwindowed version
			conff_tmp = []
			for j in range(self.cfg[0]['ext_f'].shape[0]):
				conff_tmp.append(
					(self.vars[i]['u_4f'][:,:,j]**2 + 1e-20)/\
					tf.sqrt(\
						w_bc[i][j] * self.vars[i]['u_3f'][:,:,j]**2 + \
						w_bc1[i][j] * self.vars[i]['u_4f'][:,:,j]**2 + \
						w_bc2[i][j] * self.vars[i]['u_3f'][:,:,j]*self.vars[i]['u_4f'][:,:,j] + \
						self.vars[i]['u_4f'][:,:,j]**4 + \
						1e-10\
					)
				)
			conff_tmp = tf.stack(conff_tmp, 2)

			# final confidence
			self.vars[i]['conf'] = conf_tmp
			self.vars[i]['conff'] = conff_tmp
			self.vars[i]['w_bc'] = w_bc[i]
			self.vars[i]['w_bc1'] = w_bc1[i]
			self.vars[i]['w_bc2'] = w_bc2[i]


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


		conff_align = []

		for i in range(len(self.cfg)):
			for j in range(self.cfg[0]['ext_f'].shape[0]):
				idx = i * (self.cfg[0]['ext_f'].shape[0]) + j
				tmp_align = (self.vars_align['u_4f'][:,:,idx]**2 + 1e-20)/\
					tf.sqrt(\
						w_bc[i][j] * self.vars_align['u_3f'][:,:,idx]**2 + \
						w_bc1[i][j] * self.vars_align['u_4f'][:,:,idx]**2 + \
						w_bc2[i][j] * self.vars_align['u_3f'][:,:,idx]*self.vars_align['u_4f'][:,:,idx] + \
						self.vars_align['u_4f'][:,:,idx]**4 + \
						1e-10\
					)	
				# aligned confidence
				conff_align.append(
					tmp_align
				)
			self.vars[i]['conff_align'] = conff_align[i]			

		self.vars_align['conf'] = tf.stack(conf_align,2)
		self.vars_align['conff'] = tf.stack(conff_align,2)
		return

	def w3_baseline_conf_der(self):
		# setting up derivatives for the confidence
		vars_to_add = ['w_bc', 'w_bc1', 'w_bc2']
		
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
					else:
						# only work for j+k=1
						tmp = tf.unstack(self.der[i]['dLd'+var][ef])
						tmp_1 = [tmp[0],tmp[2],tmp[1]]
						self.der[i]['dLd'+var][ef] += tf.stack(tmp_1,0)
						self.der[i]['dLd'+var][ef] /= 2
		return

	# fusion methods
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

	# align images in different res
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
				tf.concat(self.vars_align[var],3), [0]
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
				tf.concat(self.vars_align[var],3), [0]
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
				tf.concat(self.vars_align[var],3), [0]
			)
		return 

	# cut out the invalid region
	def valid_windowed_region(self):
		# cut out the bad parts
		vars_to_cut = [\
			'u_1','u_2',\
		]
		vars_to_cut3 = [\
			'u_3','u_4','Z','conf','u_3f','u_4f','Z_f','conff',\
		]

		for i in range(len(self.cfg)):
			rows_cut = int((self.cfg[i]['gauss'].shape[0]-1)/2 + (self.cfg[i]['ext_f'].shape[1]-1)/2)
			cols_cut = int((self.cfg[i]['gauss'].shape[1]-1)/2 + (self.cfg[i]['ext_f'].shape[2]-1)/2)
			rows = self.cfg[i]['szx_sensor']
			cols = self.cfg[i]['szy_sensor']
			
			for var in vars_to_cut:
				self.vars[i][var] = \
					self.vars[i][var][
						cols_cut:cols-cols_cut,
						rows_cut:rows-rows_cut
					]
			for var in vars_to_cut3:
				self.vars[i][var] = \
					self.vars[i][var][
						cols_cut:cols-cols_cut,
						rows_cut:rows-rows_cut,
						:
					]
		return 

	# cutting out invalid areas for fused data
	def valid_windowed_region_fuse(self):
		# cut out the bad part
		vars_to_cut = [\
			'Z','conf', 'Z_gt','Z_f','conff'\
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

			if var != 'Z_gt':
				self.vars_align[var] = \
					self.vars_align[var][
						cols_cut:cols-cols_cut,
						rows_cut:rows-rows_cut,
						:
					]
		return 

	# error functions
	def softmax_err(self):
		p = 1
		err_offsets = [\
			tf.reduce_mean(\
				tf.abs(\
					self.Zs[k]-self.vars_fuse['Z_gt']\
				)**p\
			)**(1/p) \
			for k in range(len(self.Zs))
		]
		err_offsets = tf.stack(err_offsets, 0)
		
		# not sure if it will cause numerical problem
		ws = tf.nn.softmax(-err_offsets*1e10)
		self.ws = ws

		# fuse the results using softmax
		err = tf.reduce_sum(err_offsets*ws)
		return err

	def softmax_err1(self):
		return self.softmax_err()

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
		self.vars_align['Zw_err_flat'] = tf.stack(
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

	def one_norm_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return tf.reduce_mean((self.vars_fuse['Z_err_flat']+0.01))

	def one_norm_err1(self):
		return self.one_norm_err()

	def one_norm_err2(self):
		return self.one_norm_err()

	def two_norm_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		# adding 0.01 is for numerical stability of the derivative
		return (tf.reduce_mean((self.vars_fuse['Z_err_flat'])**2))**0.5

	def two_norm_err1(self):
		return self.two_norm_err()

	def two_norm_err2(self):
		return self.two_norm_err()

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

	def one_round_loss(self, I, Loc, offset):
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, offset[i])

			# Update the loss
			self.update_loss()
		return self.print_loss()

	def one_step_training_force(self, I, Loc, offset, step, min_step, temperature = 0.0):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss

		# self.visual_heatmap(I, Loc, conf_thre=-np.inf)
		# self.visual_err_conf_map(I, Loc, log=False)
		# self.sparsification_map(I, Loc)
		# self.AUC_map(I, Loc)
		# pdb.set_trace()

		print("Below is a new optimization step")

		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, offset[i])

			# pdb.set_trace()

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

		self.visual_heatmap(I, Loc, offset, conf_thre=-np.inf)

		# starting to find a good step size
		# we add some simulated annealing to the function
		print("Start to find a good step size")
		self.get_old_var()
		while(step > min_step):
			print("Current step size:", step)
			self.update_apply_var(step)
			# run one round
			new_loss = self.one_round_loss(I, Loc, offset)
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

	def apply_der(self):
		# This function conduct SGD
		self.session.run(self.train[self.cur_err_func])
		return

	def one_step_training_SGD(self, I, Loc, offset, step=None, min_step=None, temperature=None):
		# perform one step training by SGD
		print("Below is a new optimization step.")
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, offset[i])

			# Update the derivative
			self.update_loss()
			self.update_der()

		print("Parameters before moving:")
		old_loss = self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# conduct optimization
		self.apply_der()
		self.clear_der()

		# self.visual_heatmap(I, Loc, offset, conf_thre=-np.inf)
		return 1, old_loss

	def visual_mean_var(self, I, Loc, conf_thre=0.):
		# since the confidence is not necessary max to 1
		# conf_thre is the threshold of the max confidence
		# input images
		Z_mean = np.zeros((len(I),), dtype = np.float32)
		Z_std = np.zeros((len(I),), dtype = np.float32)
		Z_gt = np.zeros((len(I),), dtype = np.float32)
		counter = np.zeros((len(I),), dtype = np.float32)

		for i in range(len(I)):
			# input images
			Z_gt_tmp = Loc[i,2,int((Loc.shape[2]-1)/2)]
			if Z_gt_tmp in Z_gt:
				j = np.where(Z_gt == Z_gt_tmp)[0][0]
			else:
				j = i
			Z_gt[j] = Z_gt_tmp
			counter[j] += 1
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt[j]
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z = res['Z'].flatten()
			conf = res['conf'].flatten()

			Z_mean[j] += np.mean(Z[np.where(conf>conf_thre)])
			Z_std[j] += np.std(Z[np.where(conf>conf_thre)])

		Z_mean /= counter
		Z_std /= counter
		Z_gt = Z_gt/counter *counter

		# draw the histograms
		fig = plt.figure()
		Z_gt -= self.cfg[0]['Z_0']
		Z_mean -= self.cfg[0]['Z_0']
		plt.errorbar(Z_gt, Z_mean, Z_std, linestyle='None', marker='^')
		min_depth = np.nanmin(Z_gt)
		max_depth = np.nanmax(Z_gt)
		min_depth = 0.3
		max_depth = 1.1
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.plot([min_depth, max_depth], [min_depth, max_depth])
		plt.show()
		return 

	def visual_heatmap(self, I, Loc, offset, conf_thre=0.):
		# since the confidence is not necessary max to 1
		# conf_thre is the threshold of the max confidence
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt, offset[0])

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)


		query_list_layered = ['Z', 'conf']
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

		for i in range(len(I)):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt, offset[i])

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			idx_unw += num_unw

			query_list_layered = ['Z', 'conf']
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

				idx_unw_l[j] += num_unw_l[j]

		# throw away all points with confidence smaller than conf_thre
		conf_thre = 0.999
		for j in range(len(self.cfg)):
			idx = np.where(conf_flat_layered[j] > conf_thre)
			Z_flat_layered[j] = Z_flat_layered[j][idx]
			Z_gt_flat_layered[j] = Z_gt_flat_layered[j][idx]
			conf_flat_layered[j] = conf_flat_layered[j][idx]

		# draw the histograms
		fig = plt.figure()
		for i in range(len(self.cfg)):
			self.heatmap(\
				Z_flat_layered[i]-self.cfg[0]['Z_0'], 
				Z_gt_flat_layered[i]-self.cfg[0]['Z_0'], 
				fig, 
				2,4,i+1, 
				'i='+str(i)+',conf>'+ str(conf_thre)
			)

		# # dump the prediction for quadratic calibration
		# pdb.set_trace()
		# Z = np.reshape(Z_flat,(-1,)+res['Z'].shape)
		# with open('./quad.pickle','wb') as f:
		# 	data = {
		# 		'Z':		Z,
		# 		'Loc':		Loc,
		# 		'offset':		offset,
		# 	}
		# 	# dump the data into the file
		# 	pickle.dump(data, f)

		conf_thre = [0,0.9,0.99,0.999]
		for i in range(len(conf_thre)):
			idx = np.where(conf_flat > conf_thre[i])
			Z_flat = Z_flat[idx]
			Z_gt_flat = Z_gt_flat[idx]
			conf_flat = conf_flat[idx]
			self.heatmap(\
				Z_flat-self.cfg[0]['Z_0'], 
				Z_gt_flat-self.cfg[0]['Z_0'], 
				fig, 
				2,4,i+5, 
				'Z,conf>'+ str(conf_thre[i])
			)

		plt.show()
		return 

	def visual_heatmap_percent(self, I, Loc, per_thre=0.):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		Z_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		conf_flat = np.empty((num_unw,len(I)), dtype = np.float32)

		query_list_layered = ['Z', 'conf']
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
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[:,i] = res['Z'].flatten()
			Z_gt_flat[:,i] = res['Z_gt'].flatten()
			conf_flat[:,i] = res['conf'].flatten()

			query_list_layered = ['Z', 'conf']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][:,i] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][:,i] = \
					np.ones(Z_gt_flat_layered[j][:,i].shape) * Z_gt
				conf_flat_layered[j][:,i] = \
					res_layered[j]['conf'].flatten()

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

		# draw the histograms
		fig = plt.figure()
		self.heatmap(\
			Z_flat, 
			Z_gt_flat, 
			fig, 
			1,5,1, 
			'fused_result'
		)
		for i in range(len(self.cfg)):
			self.heatmap(\
				Z_flat_layered[i], 
				Z_gt_flat_layered[i], 
				fig, 
				1,5,i+2, 
				'layer '+str(i)
			)

		plt.show()
		return 

	def visual_err_conf_map(self, I, Loc, log=True):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)


		# Initialization for recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)

		query_list_layered = ['Z', 'conf']
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
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			idx_unw += num_unw
			
			query_list_layered = ['Z', 'conf']
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

				idx_unw_l[j] += num_unw_l[j]

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
			for i in range(len(self.cfg)):
				self.error_conf_map_log(\
					Z_flat_layered[i],
					Z_gt_flat_layered[i],
					conf_flat_layered[i],
					fig,
					3,3,i+2,
					'layer '+str(i)
				)
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
			for i in range(len(self.cfg)):
				self.error_conf_map(\
					Z_flat_layered[i],
					Z_gt_flat_layered[i],
					conf_flat_layered[i],
					fig,
					3,3,i+2,
					'layer '+str(i)
				)
				

		plt.show()
		return 

	def sparsification_map(self, I, Loc):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)


		# Initialization for recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		idx_unw = 0
		Z_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*len(I),), dtype = np.float32)
		conf_flat = np.empty((num_unw*len(I),), dtype = np.float32)

		query_list_layered = ['Z', 'conf']
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
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z'].flatten()
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt'].flatten()
			conf_flat[idx_unw:idx_unw+num_unw] = res['conf'].flatten()

			idx_unw += num_unw

			query_list_layered = ['Z', 'conf']
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

				idx_unw_l[j] += num_unw_l[j]

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
		for i in range(len(self.cfg)):
			self.sparsification_plt(\
				Z_flat_layered[i],
				Z_gt_flat_layered[i],
				conf_flat_layered[i],
				fig,
				3,3,i+2,
				'layer '+str(i)
			)
			
		plt.show()
		return 

	def AUC_map(self, I, Loc):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		Z_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		conf_flat = np.empty((num_unw,len(I)), dtype = np.float32)

		query_list_layered = ['Z', 'conf']
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
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[:,i] = res['Z'].flatten()
			Z_gt_flat[:,i] = res['Z_gt'].flatten()
			conf_flat[:,i] = res['conf'].flatten()

			query_list_layered = ['Z', 'conf']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][:,i] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][:,i] = \
					np.ones(Z_gt_flat_layered[j][:,i].shape) * Z_gt
				conf_flat_layered[j][:,i] = \
					res_layered[j]['conf'].flatten()

		AUC = np.empty((Z_flat.shape[1],))
		AUC_layered = [
			np.empty((Z_flat_layered[j].shape[1],))
			for j in range(len(self.cfg))
		]

		# compute the AUC
		for i in range(len(AUC)):
			AUC[i] = self.area_under_spars_curve(
				Z_flat[:,i],
				Z_gt_flat[:,i],
				conf_flat[:,i]
			)

			for j in range(len(self.cfg)):
				AUC_layered[j][i] = self.area_under_spars_curve(
					Z_flat_layered[j][:,i],
					Z_gt_flat_layered[j][:,i],
					conf_flat_layered[j][:,i]
				)

		# plot the AUC
		fig = plt.figure()
		ax = fig.add_subplot(3,3,1, title="fused result")
		ax.plot(Z_gt_flat[0,:], AUC,'o')
		for j in range(len(self.cfg)):
			ax = fig.add_subplot(3,3,j+2, title="layer "+str(j))
			ax.plot(Z_gt_flat_layered[j][0,:], AUC_layered[j], 'o')

		plt.show()

		
		# save the data
		lpickle = len(glob.glob('./test_results/pyConfLensFlowNetFast_ext/*.pickle'))
		fileName = os.path.join(\
			'./test_results/pyConfLensFlowNetFast_ext/'+str(lpickle)+".pickle"
		)
		with open(fileName,'wb') as f:
			cfg_data = {
				'Z_flat':				Z_flat,
				'Z_gt_flat':			Z_gt_flat,
				'conf_flat':			conf_flat,
				'Z_flat_layered':		Z_flat_layered,
				'Z_gt_flat_layered':	Z_gt_flat_layered,
				'conf_flat_layered':	conf_flat_layered,
				'AUC':					AUC,
				'AUC_layered':			AUC_layered,
			}
			# dump the data into the file
			pickle.dump(cfg_data, f)

	def visual_err_percent(self, I, Loc, per_thre=0.):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I[0,:,:,0].shape) * Z_gt
		self.input_images(I[0,:,:,:], Z_map_gt)

		# Initialization of recording for faster visualization
		query_list = ['Z','Z_gt','conf']
		res = self.query_results(query_list)

		num_unw = len(res['Z'].flatten())
		Z_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw,len(I)), dtype = np.float32)
		conf_flat = np.empty((num_unw,len(I)), dtype = np.float32)

		query_list_layered = ['Z', 'conf']
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
			query_list = ['Z','Z_gt','conf']
			res = self.query_results(query_list)

			Z_flat[:,i] = res['Z'].flatten()
			Z_gt_flat[:,i] = res['Z_gt'].flatten()
			conf_flat[:,i] = res['conf'].flatten()

			query_list_layered = ['Z', 'conf']
			res_layered = self.query_results_layered(query_list_layered)
			for j in range(len(self.cfg)):
				Z_flat_layered[j][:,i] = \
					res_layered[j]['Z'].flatten()
				Z_gt_flat_layered[j][:,i] = \
					np.ones(Z_gt_flat_layered[j][:,i].shape) * Z_gt
				conf_flat_layered[j][:,i] = \
					res_layered[j]['conf'].flatten()

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

		# draw the histograms
		fig = plt.figure()
		self.heatmap(\
			Z_flat, 
			Z_gt_flat, 
			fig, 
			1,5,1, 
			'fused_result'
		)
		for i in range(len(self.cfg)):
			self.heatmap(\
				Z_flat_layered[i], 
				Z_gt_flat_layered[i], 
				fig, 
				1,5,i+2, 
				'layer '+str(i)
			)

		plt.show()
		return 

	def heatmap(self, Z_flat, Z_gt_flat, fig, s1,s2,s3, fig_name):
		# return if no points given
		if Z_flat.shape[0] == 0:
			return
		# compute average error
		err = np.mean(np.abs(Z_flat - Z_gt_flat))

		# draw the heatmap of depth prediction
		step_in_m = 0.02
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
