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
from training_pyConfLensFlowNetFast import training_pyConfLensFlowNetFast
from training_pyConfLensFlowNetFast import KEY_RANGE
import matplotlib.pyplot as plt
import matplotlib as mpl

NAN_COLOR = np.array([0,0,255])
FONT_COLOR = (0,255,0)

PADDING = 'VALID'
class pyConfLensFlowNetFast(training_pyConfLensFlowNetFast):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = None):
		#default configuraiton
		self.cfg = []
		self.cfg.append({
			# finite difference coefficient
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

		self.image_to_show = ['Z','conf','Z_gt']
		self.vars_to_fuse = ['Z','conf','u_1','u_2','u_3','u_4']

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

				if self.cfg[0]['separable']:
					wx.append(tf.constant(
						self.cfg[i]['wx'], dtype = tf.float32
					))
					wy.append(tf.constant(
						self.cfg[i]['wy'], dtype = tf.float32
					))
				else:
					w.append(tf.constant(
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
				u_4w.append(-u_2w[i] + a0[i] * u_1w[i] + 1e-5)
				Zw.append(u_3w[i] / u_4w[i] + Z_0[0])

				# unwindowed version
				u_1.append(I_lap[i])
				u_2.append(I_t[i])
				u_3.append(a0[i] * a1[0] * u_1[i])
				u_4.append(-u_2[i] + a0[i] * u_1[i] + 1e-5)
				Z.append(u_3[i] / u_4[i] + Z_0[0])
				
				#save references to required I/O]
				self.vars.append({})
				self.vars[i]['I'] = I_batch[i]
				self.vars[i]['I_lap'] = I_lap_batch[i]

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

			# align depth and confidence maps
			self.align_maps(['u_1','u_2','u_3','u_4'])

			# compute the aligned version of Z
			self.vars_align['Z'] = \
				self.vars_align['u_3'] / (self.vars_align['u_4'] + 1e-5)
			Z_align = []
			for i in range(len(self.cfg)):
				Z_align.append(self.vars_align['Z'][:,:,i] + Z_0[0])
			self.vars_align['Z'] = tf.pack(Z_align, 2)

			# compute windowed and unwindowed confidence
			eval('self.'+self.cfg[i]['conf_func']+'()')

			# cut the valid region for windowed version
			self.valid_windowed_region()

			# save the ground truth
			self.vars_fuse['Z_gt'] = Z_gt

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
				confw = 1 - tf.squeeze(separable_window(
					1-conf,
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

	# cutting out invalid areas
	def valid_windowed_region(self):
		# find out valid regions for the pyramids
		vars_to_cut = [\
			'u_1w','u_2w','u_3w','u_4w','Zw','confw',\
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

	# query results
	def regular_output(self, conf_thre=0, log = False):
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

			if (k == 'Z') and ('conf' in self.image_to_show):
				self.results[k][np.where(self.results['conf']<conf_thre)]=np.NaN

		self.cache['draw'] = tile_image(\
									I = self.results, \
									rng = rng, \
									log = log, \
									title = "Regular Output", \
								)
		cv2.imshow(self.netName, self.cache['draw'])

	def validate_pyramid(self):
		res_dict = {}
		rng = {}
		i = 1
		res_dict['I_lap'] = self.vars[i]['I_lap'][:,:,0]
		res_dict['I'] = self.vars[i]['I'][:,:,0]
		res_dict['Z'] = self.vars[i]['Z']

		self.results = self.session.run(res_dict)

		# self.results['I_lap'] = self.results['I_lap'][2:-2,2:-2]
		# self.results['I'] = self.results['I'][2:-2,2:-2]
		# self.results['Z'] = self.results['Z'][2:-2,2:-2]
		rng['I_lap'] = [np.NaN, np.NaN]
		rng['I'] = [np.NaN, np.NaN]
		rng['Z'] = [-1.0,-0.25]

		self.cache['draw'] = tile_image(\
									I = self.results, \
									rng = rng, \
									log = False, \
									title = "Regular Output", \
								)
		cv2.imshow(self.netName, self.cache['draw'])