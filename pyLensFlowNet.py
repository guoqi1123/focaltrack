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
import matplotlib.pyplot as plt
import scipy.misc

NAN_COLOR = np.array([0,0,255])
FONT_COLOR = (0,255,0)

PADDING = 'VALID'
class pyLensFlowNet():
	def __init__(self):
		self.netName = 'pyLensFlowNet'
		self.bNets = []

	def add_basicNet(self, cfg):
		self.bNets.append(
			basicPyLensFlowNet(cfg)
		)

	def heatmap(self, Z_flat, Z_gt_flat, num_depth, fig, s1,s2,s3, fig_name):
		# compute average error
		err = np.mean(np.abs(Z_flat - Z_gt_flat))

		# draw the heatmap of depth prediction
		step_in_m = 0.01
		min_depth = self.bNets[0].cfg['wr'][0]-step_in_m/2
		max_depth = self.bNets[0].cfg['wr'][1]+step_in_m/2
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
		num_inst = int(np.rint(len(Z_flat)/num_depth))
		pdb.set_trace()
		htmap /= num_inst
		extent = [\
			xedgs[0], 
			xedgs[-1], 
			yedgs[0], 
			yedgs[-1]
		]
		# since the batch is in equal size, we could directly use imshow
		title = fig_name+", err: "+str(err)
		ax = fig.add_subplot(s1,s2,s3, title=title)
		plt.imshow(htmap, interpolation='bilinear', origin='low', extent=extent)
		plt.colorbar()

		plt.plot([min_depth, max_depth], [min_depth, max_depth], 'k')

		return

	def depthMap(self, I, I_lap_batch):
		# this function inputs the Gaussian and Laplacian
		# pyramid of ONE image, and return the depth map
		I_shape = (I[0].shape[0],I[0].shape[1])
		self.Z_map = np.zeros((len(I),)+I_shape)
		self.conf_map = np.zeros((len(I),)+I_shape, dtype=np.float32)
		self.Z_layered = []
		self.Zw_layered = []

		for i in range(len(self.bNets)):
			self.bNets[i].input_images(\
				I[i],\
				I_lap_batch[i],\
			)
			query_list = ['Z_valid','conf_valid','Zw_valid']
			res = self.bNets[i].query_results(query_list)
			self.Z_layered.append(res['Z_valid'])
			self.Zw_layered.append(res['Zw_valid'])

			# obtain results and resize it
			Z_map_temp = cv2.resize(
				res['Z_valid'], 
				(I[0].shape[1],I[0].shape[0]),
				interpolation= cv2.INTER_NEAREST
			).astype(np.float32)

			conf_map_temp = cv2.resize(
				res['conf_valid'], 
				(I[0].shape[1],I[0].shape[0]),
				interpolation= cv2.INTER_LINEAR
			).astype(np.float32)
			
			# find the place out of working range
			# and set the confidence to be zero
			conf_map_temp[\
				Z_map_temp < self.bNets[i].cfg['wr'][0]] = 0
			conf_map_temp[\
				Z_map_temp > self.bNets[i].cfg['wr'][1]] = 0

			self.Z_map[i,:,:] = Z_map_temp
			self.conf_map[i,:,:] = conf_map_temp

		# fuse the Z_map according to the confidence
		max_conf = np.argmax(self.conf_map,axis=0)
		xx, yy = np.meshgrid(
			np.arange(I_shape[1]),
			np.arange(I_shape[0]),
		)
		idx = np.stack((max_conf, yy, xx), axis=0)
		idx = np.reshape(idx, (3,-1))
		idx = (idx[0,:], idx[1,:], idx[2,:])
		Z = self.Z_map[idx]
		self.Z = np.reshape(Z, I_shape).astype(np.float32)

		return self.Z_map, self.conf_map, self.Z

	def display(self):
		self.image_to_show = ['Z','Z_gt']
		res = {}
		for k in self.image_to_show:
			res[k] = eval('self.'+ k)

		rng = {}
		for k in self.image_to_show:
			if k in KEY_RANGE.keys():
				rng[k] = KEY_RANGE[k]
			else:
				rng[k] = [np.NaN, np.NaN]

		self.cache = tile_image(\
									I = res, \
									rng = rng, \
									log = False, \
									title = "Regular Output", \
								)
		cv2.imshow(self.netName, self.cache)
		cv2.waitKey(1)

	def compute_err(self, I, I_lap_batch, Z_gt):
		# Z_gt should be in the same size as the largest I
		self.Z_gt = Z_gt.astype(np.float32)
		self.depthMap(I, I_lap_batch)
		self.err_Z = np.mean(np.abs(self.Z - Z_gt))
		self.err_Z_map = np.mean(np.abs(self.Z_map - Z_gt), axis=(1,2))

		print("Fused depth map has error: ", self.err_Z)
		print("Unfused depth map has error: ", self.err_Z_map)

		# also we compute the error variance
		mean_Z = np.mean(self.Z)
		Z_flat = np.reshape(self.Z,[-1])
		data_low = Z_flat[Z_flat < mean_Z]
		data_high = Z_flat[Z_flat >= mean_Z]
		
		self.mean_Z = mean_Z
		self.std_low = np.sqrt(np.mean((data_low - mean_Z)**2))
		self.std_high = np.sqrt(np.mean((data_high - mean_Z)**2))

		self.display()
		return self.Z, self.Z_layered, self.Zw_layered

class basicPyLensFlowNet(focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'ft': np.array([[[-0.5,0.5]]]),
			'fave': np.array([[[0.5,0.5]]]),
			'a0' : 1, #sensor distance
			'a1' : 1, #focal distance
			'Z_0': 0,
			# convolution window
			'separable' : True, # indicator of separability of conv
			'w' : np.ones((35,35)), # unseparable window
			'wx': np.ones((1,35)), # separable window
			'wy': np.ones((35,1)), # separable window
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
		self.input_var = {}
		self.loss = 0
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

		self.image_to_show = ['Z_valid','Z_gt_valid']
		self.netName = "Training Lens Flow"

	"""imports a batch of frame into """
	def input_images(self, I, I_lap_batch):
		# import data into the network
		input_dict = {
			self.I_in: I,
			self.I_lap_batch: I_lap_batch,
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
			I_init = np.zeros(
				self.resolution+(self.cfg['ft'].shape[2],), 
				dtype = np.float32
			)
			self.I_in = tf.Variable(I_init)
			self.I_lap_batch = tf.Variable(I_init)
			I = tf.Variable(I_init)
			I_lap_batch = tf.Variable(I_init)

			I_trans = tf.transpose(I, perm=[2,0,1])		

			"""Input parameters"""
			a0 = tf.constant(self.cfg['a0'], dtype = tf.float32)
			a1 = tf.constant(self.cfg['a1'], dtype = tf.float32)

			ft = tf.constant(self.cfg['ft'], dtype = tf.float32)
			fave = tf.constant(self.cfg['fave'], dtype = tf.float32)

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

			# Generate the differential images
			I_t = dIdt(I, ft)

			# Generate the Laplacian
			I_lap = dIdt(I_lap_batch, fave)

			# windowed version 
			if self.cfg['separable']:
				u_1w = separable_window(
					I_lap*I_lap, 
					wx, 
					wy, 
					PADDING,
				)
				u_2w = separable_window(
					I_t*I_lap, 
					wx, 
					wy, 
					PADDING,
				)
			else:
				u_1w = unseparable_window(
					I_lap*I_lap, 
					w, 
					PADDING,
				)
				u_2w = unseparable_window(
					I_t*I_lap, 
					w,
					PADDING,
				)
			u_1w = tf.squeeze(u_1w, [0,3])
			u_2w = tf.squeeze(u_2w, [0,3])
			u_3w = a0 * a1 * u_1w
			u_4w = -u_2w + a0 * u_1w + 1e-10
			Zw = u_3w / u_4w + self.cfg['Z_0']
			uncw = tf.sqrt(u_3w**2 + u_4w**2 + u_4w**4+1e-10)/(u_4w**2+1e-20)
			confw = 1/uncw

			# unwindowed version
			u_1 = I_lap * I_lap
			u_2 = I_t * I_lap
			u_3 = a0 * a1 * u_1
			u_4 = -u_2 + a0 * u_1 + 1e-10
			Z = u_3 / u_4 + self.cfg['Z_0']
			unc = tf.sqrt(u_3**2 + u_4**2 + u_4**4+1e-10)/(u_4**2+1e-20)
			conf = 1/unc
			
			#save references to required I/O]
			self.vars['I'] = I
			
			# windowed version 
			self.vars['u_1w'] = u_1w
			self.vars['u_2w'] = u_2w
			self.vars['u_3w'] = u_3w
			self.vars['u_4w'] = u_4w

			self.vars['Zw'] = Zw
			self.vars['uncw'] = uncw
			self.vars['confw'] = confw
			
			# unwindowed version
			self.vars['u_1'] = u_1
			self.vars['u_2'] = u_2
			self.vars['u_3'] = u_3
			self.vars['u_4'] = u_4

			self.vars['Z'] = Z
			self.vars['unc'] = unc
			self.vars['conf'] = conf
	
			#save a reference for easy debugging purposes - 
			#there are some automatic ways of pulling this 
			#data out of the graph but this is much easier 
			#when prototyping
			self.vars['I_t'] = I_t
			self.vars['I_lap'] = I_lap

			self.valid_region()

			#add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				I_lap_batch.assign(self.I_lap_batch),
			)

			#do not add anything to the compute graph after this line
			init_op = tf.initialize_all_variables()
			self.session.run(init_op)

	"""Here we define several error functions"""
	def valid_region(self):
		# find out valid regions for windowed parameters
		vars_to_cut = [\
			'u_1w','u_2w','u_3w','u_4w','Zw','uncw','confw',\
		]
		if PADDING == 'SAME':
			if self.cfg['separable']:
				rows_cut = int(
					(self.cfg['wx'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['wy'].shape[0]-1)/2
				)
			else:
				rows_cut = int(
					(self.cfg['w'].shape[1]-1)/2
				)
				cols_cut = int(
					(self.cfg['w'].shape[0]-1)/2
				)
			cols = self.cfg['szy_sensor']
			rows = self.cfg['szx_sensor']
		elif PADDING == 'VALID':
			rows_cut = 0
			cols_cut = 0
			cols = self.cfg['szy_sensor'] - \
				(self.cfg['wy'].shape[0]-1)
			rows = self.cfg['szx_sensor'] - \
				(self.cfg['wx'].shape[1]-1)
		
		for var in vars_to_cut:
			self.vars[var+'_valid'] = self.vars[var][
				cols_cut:cols-cols_cut,
				rows_cut:rows-rows_cut
			]
			self.vars[var+'_valid_flat'] = tf.reshape(
				self.vars[var+'_valid'], [-1]
			)

		# cut unwindowed variables
		vars_to_cut = [\
			'u_1','u_2','u_3','u_4','Z','unc','conf'
		]
		rows_cut = 0
		cols_cut = 0
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
