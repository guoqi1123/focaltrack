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

NAN_COLOR = np.array([0,0,255])
FONT_COLOR = (0,255,0)

PADDING = 'VALID'
class training_pyLensFlowNet():
	def __init__(self):
		self.bNets = []

	def add_basicNet(self, cfg):
		self.bNets.append(
			training_basicPyLensFlowNet(cfg)
		)

class training_basicPyLensFlowNet(training_focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
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
			'wx': np.ones((1,35)), # separable window
			'wy': np.ones((35,1)), # separable window
			# other parameters
			'szx_sensor': 200,
			'szy_sensor': 200,
			'outs': ['Z'],
			'learn_rate': 0.01,
			'err_func': ['huber_err']
		}

		# Change configurations according to the input
		for k in cfg.keys():
			self.cfg[k] = cfg[k]

		self.cfg['P'] = 1/self.cfg['f']

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
		self.cache = {}
		self.graph = tf.Graph()
		self.session = tf.Session(graph = self.graph)
		self.build_graph()

		self.image_to_show = ['Z_valid','Z_gt_valid','u_1','u_2']
		self.netName = "Training Lens Flow"

	"""imports a batch of frame into """
	def input_images(self, I, I_lap_batch, Z):
		# import data into the network
		input_dict = {
			self.I_in: I,
			self.Z_in: Z,
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
			Z_init = np.zeros(
				self.resolution,
				dtype = np.float32
			)
			self.I_in = tf.Variable(I_init)
			self.I_lap_batch = tf.Variable(I_init)
			self.Z_in = tf.Variable(Z_init)
			I = tf.Variable(I_init)
			I_lap_batch = tf.Variable(I_init)

			Z_gt = tf.Variable(Z_init)
			Z_0 = tf.Variable(self.cfg['Z_0'])

			I_trans = tf.transpose(I, perm=[2,0,1])		

			"""Input parameters"""
			a0 = tf.Variable(self.cfg['a0'], dtype = tf.float32)
			a1 = tf.Variable(self.cfg['a1'], dtype = tf.float32)

			ft = tf.Variable(self.cfg['ft'], dtype = tf.float32)
			fave = tf.Variable(self.cfg['fave'], dtype = tf.float32)

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
			u_4w = -u_2w + a0 * u_1w + 1e-6
			Zw = u_3w / u_4w + Z_0
			uncw = tf.sqrt(u_3w**2 + u_4w**2 + u_4w**4+1e-10)/(u_4w**2+1e-20)
			confw = 1/uncw

			# unwindowed version
			u_1 = I_lap * I_lap
			u_2 = I_t * I_lap
			u_3 = a0 * a1 * u_1
			u_4 = -u_2 + a0 * u_1 + 1e-6
			Z = u_3 / u_4 + Z_0
			unc = tf.sqrt(u_3**2 + u_4**2 + u_4**4+1e-10)/(u_4**2+1e-20)
			conf = 1/unc
			
			#save references to required I/O]
			self.vars['I'] = I
			self.vars['Z_gt'] = Z_gt
			
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
			
			####################################################################
			# below are derivatives and optimization
			####################################################################
			self.der['dLdft'] = {}
			self.vars['ft'] = ft			

			self.der['dLdfave'] = {}
			self.vars['fave'] = fave

			self.der['dLda0'] = {}
			self.der['dLda1'] = {}
			self.der['dLdZ_0'] = {}
			self.vars['a0'] = a0
			self.vars['a1'] = a1
			self.vars['Z_0'] = Z_0

			# compute error
			self.vars['loss_func'] = {}
			for ef in self.cfg['err_func']:
				self.vars['loss_func'][ef] = \
					eval('self.'+ef+'()')

			for ef in self.cfg['err_func']:
				dLdft, dLdfave = \
					tf.gradients( \
						self.vars['loss_func'][ef], [ft, fave]
				)
				num_inst = tf.to_float(\
					tf.shape(self.vars['loss_func'][ef])[0]
				)				

				# restrict the symmetry:
				# dLdft + tf.reverse(dLdft) = 0
				dLdft = dLdft - tf.reverse(dLdft, dims = [False, False, True])
				dLdft = dLdft + tf.reverse(dLdft, dims = [True, False, False])\
				 	+ tf.reverse(dLdft, dims = [False, True, False])\
				 	+ tf.reverse(dLdft, dims = [True, True, False])
				dLdft = dLdft + tf.transpose(dLdft, perm = [1,0,2])
				dLdft = dLdft + tf.reverse(dLdft, dims = [True, False, False])
				dLdft = dLdft + tf.reverse(dLdft, dims = [False, True, False])
				dLdft = dLdft / 64. / num_inst
				self.der['dLdft'][ef] = dLdft
				

				# derivative w.r.t. to fave
				dLdfave = dLdfave / num_inst
				self.der['dLdfave'][ef] = dLdfave

				#### derivative w.r.t. optical parameters
				dLda0, dLda1, dLdZ_0 = tf.gradients(
					self.vars['loss_func'][ef], [a0, a1, Z_0]
				)
				if dLda0 == None:
					dLda0 = tf.constant(0,dtype =tf.float32)
				if dLda1 == None:
					dLda1 = tf.constant(0,dtype =tf.float32)
				if dLdZ_0 == None:
					dLdZ_0 = tf.constant(0,dtype =tf.float32)

				dLda0 = dLda0 / num_inst * self.cfg['da0_ratio']
				dLda1 = dLda1 / num_inst * self.cfg['da1_ratio']
				dLdZ_0 = dLdZ_0 / num_inst * self.cfg['dZ_0_ratio']
				self.der['dLda0'][ef] = dLda0
				self.der['dLda1'][ef] = dLda1
				self.der['dLdZ_0'][ef] = dLdZ_0
			

			self.output_grads = {}
			for ef in self.cfg['err_func']:
				self.output_grads[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.output_grads[ef][key] = self.der[key][ef]
				
			self.output_vars = {} # for variable updating
			for ef in self.cfg['err_func']:
				self.output_vars[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.output_vars[ef][key] = \
						self.vars[self.var_dict[key]]
			
			#### put the used derivative into a list, depending 
			# on the cfg
			grads_and_vars = {}
			for ef in self.cfg['err_func']:
				grads_and_vars[ef] = [
					der_pair[key] \
					for key in self.cfg['der_var'][ef]
				]
			self.grads_and_vars = grads_and_vars

			#### optimize using AdagradOptimizer
			opt = {}
			self.train = {}
			for ef in self.cfg['err_func']:
				opt[ef] = tf.train.AdagradOptimizer(\
					learning_rate=self.cfg['learn_rate']
				)
				self.train[ef] = opt[ef].apply_gradients(
					grads_and_vars[ef]
				)


			#save a reference for easy debugging purposes - 
			#there are some automatic ways of pulling this 
			#data out of the graph but this is much easier 
			#when prototyping
			self.vars['I_t'] = I_t
			self.vars['I_lap'] = I_lap

			#add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				I_lap_batch.assign(self.I_lap_batch),
				Z_gt.assign(self.Z_in),
			)

			# assign final derivatives			
			self.assign_der_f = {}
			for ef in self.cfg['err_func']:
				self.assign_der_f[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.assign_der_f[ef][key] = \
						self.der_f[key].assign(\
							self.input_der[key]
						)

			# assign modified variables
			self.assign_new_var = {}
			for ef in self.cfg['err_func']:
				self.assign_new_var[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.assign_new_var[ef][key] = \
						self.vars[self.var_dict[key]].assign(
							self.input_var[key]
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
			'u_1','u_2','u_3','u_4','Z','unc','conf','Z_gt'
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


		# cut for windowed 'Z_gt'
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
		self.vars['Z_gtw_valid'] = self.vars['Z_gt'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['Z_gtw_valid_flat'] = tf.reshape(
			self.vars['Z_gtw_valid'], [-1]
		)

		return 

	def compute_err_34(self):
		# compute err using u3 and u4
		self.valid_region()
		u_len = tf.sqrt(
			tf.abs(self.vars['u_4_valid'])**2+\
			tf.abs(self.vars['u_3_valid'])**2
		)
		# # move the points to be uniform length
		# u_3_uni = (self.vars['u_3_valid']+1e-5) / u_len
		# u_4_uni = (self.vars['u_4_valid']+1e-5) / u_len
		# u_3gt_uni = (u_3_uni + u_4_uni) / (1+self.vars['Z_gt_valid'])
		# u_4gt_uni = self.vars['Z_gt_valid'] * u_3gt_uni

		# self.vars['Z_err_valid'] = tf.sqrt(
		# 	(u_3_uni - u_3gt_uni)**2 + \
		# 	(u_4_uni - u_4gt_uni)**2
		# )

		z = self.vars['Z_gt_valid']
		u_3 = self.vars['u_3_valid']
		u_4 = self.vars['u_4_valid']
		d = np.sqrt(2) * tf.abs(z/(1+z)*u_3 - 1/(1+z)*u_4)
		ep = 1e-5

		self.vars['Z_err_valid'] = (d+ep)/(u_len+ep**2)

		self.vars['Z_err_valid_flat'] = tf.reshape(
			self.vars['Z_err_valid'],[-1]
		)
		return 

	def compute_err_12(self):
		# compute err using u1 and u2
		self.valid_region()
		u_len = tf.sqrt(
			tf.abs(self.vars['u_2_valid'])**2+\
			tf.abs(self.vars['u_1_valid'])**2
		)

		z = self.vars['Z_gt_valid']
		a0 = self.vars['a0']
		a1 = self.vars['a1']
		k = (a0*z - a0*a1)/z
		u_1 = self.vars['u_1_valid']
		u_2 = self.vars['u_2_valid']

		d = np.sqrt(2) * tf.abs(k/(1+k)*u_1 - 1/(1+k)*u_2)
		ep = 1e-3

		# # weight the error by inverse of length
		# self.vars['Z_err_valid'] = (d+ep)/(u_len+ep**2)

		# don't reweight the err
		self.vars['Z_err_valid'] = d

		self.vars['Z_err_valid_flat'] = tf.reshape(
			self.vars['Z_err_valid'],[-1]
		)
		return 

	def compute_err(self):
		self.valid_region()
		self.vars['Z_err_valid'] = tf.abs(
			self.vars['Zw_valid'] - self.vars['Z_gtw_valid']
		)
		self.vars['Z_err_valid_flat'] = tf.reshape(
			self.vars['Z_err_valid'],[-1]
		)
		return

	def half_norm_err_1(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		# use this to fix a bug in the code
		self.compute_err()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return tf.sqrt(self.vars['Z_err_valid_flat']+0.01)

	def one_round_loss(self, I, I_lap_batch, Loc):
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(\
				I[i,:,:,:],\
				I_lap_batch[i,:,:,:],\
				Z_map_gt\
			)

			# Update the loss
			self.update_loss()
		return self.print_loss()

	def one_step_training_force(self, I, I_lap_batch, Loc, step, min_step):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss
		print("Below is a new optimization step")
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(\
				I[i,:,:,:], \
				I_lap_batch[i,:,:,:], \
				Z_map_gt\
			)

			# show the depth map
			self.regular_output()
			cv2.waitKey(1)

			# # Query some results for drawing
			query_list = [\
				'Zw_valid_flat', \
				'Z_gtw_valid_flat',\
			]
			res = self.query_results(query_list)

			# Update the derivative
			self.update_loss()
			self.update_der()
			
		print("Parameters before moving:")
		old_loss = self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# visual analysis 
		self.visual_analysis(I, I_lap_batch, Loc)

		# starting to find a good step size
		print("Start to find a good step size")
		self.get_old_var()
		while(step > min_step):
			print("Current step size:", step)
			self.update_apply_var(step)
			# run one round
			new_loss = self.one_round_loss(I, I_lap_batch, Loc)
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

	def visual_analysis(self, I, I_lap_batch, Loc):
		# input images
		Z_gt = Loc[0,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
		self.input_images(I[0,:,:,:], I_lap_batch[0,:,:,:], Z_map_gt)


		# Initialization for recording for faster visualization
		query_list = ['Z_valid_flat','Z_gt_valid_flat','Zw_valid_flat','Z_gtw_valid_flat']
		res = self.query_results(query_list)

		num_unw = len(res['Z_valid_flat'])
		idx_unw = 0
		Z_flat = np.empty((num_unw*I.shape[0],), dtype = np.float32)
		Z_gt_flat = np.empty((num_unw*I.shape[0],), dtype = np.float32)

		num_w = len(res['Zw_valid_flat'])
		idx_w = 0
		Zw_flat = np.empty((num_w*I.shape[0],), dtype = np.float32)
		Z_gtw_flat = np.empty((num_w*I.shape[0],), dtype = np.float32)
		confw_flat = np.empty((num_w*I.shape[0],), dtype = np.float32)

		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], I_lap_batch[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output(conf_thre=0.95)
			# cv2.waitKey(1)

			# Query some results for analysis, concatenate results
			# for all images into a long vector
			query_list = ['Z_valid_flat','Z_gt_valid_flat','Zw_valid_flat','Z_gtw_valid_flat']
			res = self.query_results(query_list)

			Z_flat[idx_unw:idx_unw+num_unw] = res['Z_valid_flat']			
			Z_gt_flat[idx_unw:idx_unw+num_unw] = res['Z_gt_valid_flat']

			Zw_flat[idx_w:idx_w+num_w] = res['Zw_valid_flat']
			Z_gtw_flat[idx_w:idx_w+num_w] = res['Z_gtw_valid_flat']

			idx_unw += num_unw
			idx_w += num_w

		# draw the histograms
		fig = plt.figure()
		self.heatmap(\
			Z_flat, 
			Z_gt_flat, 
			fig, 
			1,2,1, 
			'fused_result'
		)
		self.heatmap(\
			Zw_flat, 
			Z_gtw_flat, 
			fig, 
			1,2,2, 
			'fused_result, window '+str(self.cfg['w'].shape)
		)
		plt.show()

	def find_wr(self, I, I_lap_batch, Loc):
		# this function finds the working range using the given data
		# working range is defined by the depth range with 5% relative error
		# using windowed data

		# average relative error
		self.err_thre = 0.05
		Z_gts = Loc[:,2,int((Loc.shape[2]-1)/2)]
		Z_gts_uniq = np.unique(Z_gts)
		mean_rel_err = np.zeros(\
			(Z_gts_uniq.shape[0],3),dtype = np.float32
		)
		mean_rel_err[:,0] = Z_gts_uniq

		for i in range(I.shape[0]):
			# input images
			Z_gt = Z_gts[i]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(\
				I[i,:,:,:], \
				I_lap_batch[i,:,:,:], \
				Z_map_gt\
			)

			# Query results to compute relative error
			query_list = [\
				'Zw_valid_flat', 'Z_gtw_valid_flat', 'Z_0'\
			]
			res = self.query_results(query_list)

			# find if there is already 
			find_loc = np.where(\
				mean_rel_err[:,0] == res['Z_gtw_valid_flat'][0]
			)

			mean_rel_err[find_loc,1] += np.sum(\
				np.abs(res['Zw_valid_flat']-res['Z_gtw_valid_flat'])\
				/np.abs(res['Z_gtw_valid_flat'] - res['Z_0'])\
			)

			mean_rel_err[find_loc,2] += res['Zw_valid_flat'].shape[0]
		# compute the average 
		mean_rel_err[:,1] /= mean_rel_err[:,2]

		flag = np.where(mean_rel_err[:,1] < self.err_thre)
		if flag[0].shape[0] == 0:
			Z_gt_min = 999999999999
			Z_gt_max = -999999999999
		else:
			Z_gt_min = mean_rel_err[flag,0].min()
			Z_gt_max = mean_rel_err[flag,0].max()
		self.cfg['wr'] = (Z_gt_min, Z_gt_max)

		return 