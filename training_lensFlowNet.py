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
class training_lensFlowNet(training_focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'lap': [[0,1,0],[1,-4,1],[0,1,0]],
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
			'lap':'dLdlap',
			'fave':'dLdfave',
			'ft':'dLdft',
			'a0':'dLda0',
			'a1':'dLda1',
			'Z_0':'dLdZ_0',
		}
		self.var_dict = {
			'dLdlap':'lap',
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

		self.image_to_show = ['Z_valid','Z_gt_valid']
		self.netName = "Training Lens Flow"

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

			"""Input parameters"""
			a0 = tf.Variable(self.cfg['a0'], dtype = tf.float32)
			a1 = tf.Variable(self.cfg['a1'], dtype = tf.float32)

			ft = tf.Variable(self.cfg['ft'], dtype = tf.float32)
			lap = tf.Variable(self.cfg['lap'], dtype = tf.float32)
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

			# we compute the laplacian of the image in a batch
			I_lap_batch = dIdx_batch(I_trans, lap)
			# conduct averaging
			I_lap_batch_trans = tf.transpose(I_lap_batch, perm = [1,2,0])
			I_lap = dIdt(I_lap_batch_trans, fave)

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
			
			Z = u_3w / u_4w
			unc = tf.sqrt(u_3w**2 + u_4w**2 + u_4w**4+1e-10)/(u_4w**2+1e-20)
			conf = 1/unc

			u_1 = I_lap
			u_2 = I_t
			u_3 = a0 * a1 * u_1
			u_4 = -u_2 + a0 * u_1
			
			#save references to required I/O]
			self.vars['I'] = I
			self.vars['Z'] = Z
			self.vars['unc'] = unc
			self.vars['conf'] = conf
			self.vars['Z_gt'] = Z_gt0
			self.vars['u_1w'] = u_1w
			self.vars['u_2w'] = u_2w
			self.vars['u_3w'] = u_3w
			self.vars['u_4w'] = u_4w
			self.vars['u_1'] = u_1
			self.vars['u_2'] = u_2
			self.vars['u_3'] = u_3
			self.vars['u_4'] = u_4

			####################################################################
			# below are derivatives and optimization
			####################################################################
			self.der['dLdlap'] = {}
			self.vars['lap'] = lap

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
				dLdlap, dLdft, dLdfave = \
					tf.gradients(self.vars['loss_func'][ef], [lap, ft, fave])
				num_inst = tf.to_float(tf.shape(self.vars['loss_func'][ef])[0])
				# restrict the symmetry:
				# ALSO sum(dLdlap) = 0
				# this is validated by flipping the image will flip the derivative
				# moreover it makes the derivative independent of brightness level
				# it seems that the original derivative has a constant added to it
				# which deals with the brightness level
				dLdlap = dLdlap + tf.reverse(dLdlap, dims = [True, False])\
				 		+ tf.reverse(dLdlap, dims = [False, True])\
				 		+ tf.reverse(dLdlap, dims = [True, True])
				dLdlap = dLdlap + tf.transpose(dLdlap)
				dLdlap = dLdlap + tf.reverse(dLdlap, dims = [True, False])
				dLdlap = dLdlap + tf.reverse(dLdlap, dims = [False, True])
				dLdlap = dLdlap / 32. / num_inst
				dLdlap = dLdlap - tf.reduce_mean(dLdlap)

				self.der['dLdlap'][ef] = dLdlap
				

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
			self.vars['I_t'] = I_t
			self.vars['I_lap'] = I_lap

			#add values
			self.input_data = tf.group(
				I.assign(self.I_in),
				Z_gt.assign(self.Z_in),
			)

			# assign final derivatives			
			self.assign_der_f = {}
			for ef in self.cfg['err_func']:
				self.assign_der_f[ef] = {}
				for key in self.cfg['der_var'][ef]:
					self.assign_der_f[ef][key] = \
						self.der_f[key].assign(self.input_der[key])

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
		# find out valid regions for Z, Z_gt
		vars_to_cut = ['u_1w','u_2w','u_3w','u_4w','Z','unc','conf']
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
			rows = self.cfg['szx_sensor']
			cols = self.cfg['szy_sensor']
		elif PADDING == 'VALID':
			rows_cut = int(
				(self.cfg['lap'].shape[1]-1)/2
			)
			cols_cut = int(
				(self.cfg['lap'].shape[0]-1)/2
			)
			rows = self.cfg['szx_sensor'] - (self.cfg['wx'].shape[1]-1)
			cols = self.cfg['szy_sensor'] - (self.cfg['wy'].shape[0]-1)
		
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
		rows = self.cfg['szx_sensor']
		cols = self.cfg['szy_sensor']
		self.vars['Z_gt_valid'] = self.vars['Z_gt'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
		self.vars['Z_gt_valid_flat'] = tf.reshape(
			self.vars['Z_gt_valid'], [-1]
		)

		# cut u_1, u_2,
		vars_to_cut = ['u_1','u_2','u_3','u_4']
		rows_cut = int(
			(self.cfg['lap'].shape[1]-1)/2
		)
		cols_cut = int(
			(self.cfg['lap'].shape[0]-1)/2
		)
		rows = self.cfg['szx_sensor']
		cols = self.cfg['szy_sensor']
		
		for var in vars_to_cut:
			self.vars[var+'_valid'] = self.vars[var][
				cols_cut:cols-cols_cut,
				rows_cut:rows-rows_cut
			]
			self.vars[var+'_valid_flat'] = tf.reshape(
				self.vars[var+'_valid'], [-1]
			)

		return 

	def compute_eig(self):
		self.alid_region()

		u = tf.pack(\
			[self.vars['u_1_valid_flat']/tf.reduce_sum(tf.abs(self.vars['lap'])),\
			self.vars['u_2_valid_flat']/tf.reduce_sum(tf.abs(self.vars['ft']))],
			axis = 1
		) 
		uTu = tf.matmul(u, u, transpose_a=True)

		l1 = (uTu[0,0]+uTu[1,1]+tf.sqrt((uTu[0,0]-uTu[1,1])**2 + 4*uTu[0,1]**2))/2
		l2 = (uTu[0,0]+uTu[1,1]-tf.sqrt((uTu[0,0]-uTu[1,1])**2 + 4*uTu[0,1]**2))/2

		self.vars['eig'] = tf.pack([l1,l2],axis=0)
		return

	def eig_ratio_err(self):
		# Compute the one norm of the error for each pixel
		# sparse norms are not suitable for gradient descent
		self.compute_eig()
		# no need to sum them together since the auto-differentiation 
		# will do the sum
		return tf.expand_dims(self.vars['eig'][1]/self.vars['eig'][0], 0)

	def one_step_training_force(self, I, Loc, step, min_step):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss
		print("Below is a new optimization step")
		draw_list_ave = np.empty((0,2),dtype = np.float32)
		draw_list_std = np.empty((0,2),dtype = np.float32)
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# # Query some results for drawing
			query_list = ['Z_valid_flat', 'Z_gt_valid_flat']
			res = self.query_results(query_list)

			# conf = res['conf_valid_flat']
			# thre = 0.4

			# # show how u_1 and u_2 are distributed 
			# plt.figure()
			# plt.subplot(221)
			# plt.plot(res['u_1_valid_flat'][conf>thre],res['u_2_valid_flat'][conf>thre],'ro')
			# plt.xlabel('u_1')
			# plt.ylabel('u_2')

			# plt.subplot(222)
			# plt.plot(res['u_3_valid_flat'][conf>thre],res['u_4_valid_flat'][conf>thre],'ro')
			# plt.xlabel('u_3')
			# plt.ylabel('u_4')

			# plt.subplot(223)
			# plt.hist(res['u_1_valid_flat'][conf>thre],bins=100)
			# plt.ylabel('u_1 number')
			# plt.xlabel('u_1 frequency distribution')

			# plt.subplot(224)
			# plt.hist(res['u_2_valid_flat'][conf>thre],bins=100)
			# plt.ylabel('u_2 number')
			# plt.xlabel('u_2 frequency distribution')
			# plt.show()

			# Update the derivative
			self.update_loss()
			self.update_der()
			mean_Z = np.mean(res['Z_valid_flat'])
			mean_Z_gt = np.mean(res['Z_gt_valid_flat'])
			draw_list_ave = np.concatenate(
				(draw_list_ave,
					np.column_stack((mean_Z_gt,mean_Z))
				),axis = 0
			)
			data_low = res['Z_valid_flat'][res['Z_valid_flat'] < mean_Z]
			data_high = res['Z_valid_flat'][res['Z_valid_flat']>= mean_Z]
			std_low = np.sqrt(np.mean((data_low - mean_Z)**2))
			std_high = np.sqrt(np.mean((data_high - mean_Z)**2))
			draw_list_std = np.concatenate(
				(draw_list_std,
					np.column_stack((std_low,std_high))
				),axis = 0
			)
		print("Parameters before moving:")
		old_loss = self.print_loss()
		self.finalize_der()
		self.print_grads_and_vars()

		# draw the result before training
		min_depth = draw_list_ave[:,0].min()
		max_depth = draw_list_ave[:,0].max()
		plt.figure()
		plt.plot([min_depth, max_depth], [min_depth, max_depth])	
		plt.errorbar(draw_list_ave[:,0],draw_list_ave[:,1],\
			 yerr=[draw_list_std[:,0],draw_list_std[:,1]],fmt='ro')
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.ylabel('Estimated depth (m)')
		plt.xlabel('True depth (m)')
		
		plt.show()

		# starting to find a good step size
		print("Start to find a good step size")
		self.get_old_var()
		while(step > min_step):
			print("Current step size:", step)
			self.update_apply_var(step)
			# run one round
			new_loss = self.one_round_loss(I, Loc)
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

	def one_step_training_SGD(self, I, Loc):
		# perform one step training by SGD
		print("Below is a new optimization step.")
		draw_list_ave = np.empty((0,2),dtype = np.float32)
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# Query some results for drawing
			query_list = ['Z_valid_flat','Z_gt_valid_flat']
			res = self.query_results(query_list)
			# Update the derivative
			self.update_loss()
			self.update_der()
			draw_list_ave = np.concatenate(
				(draw_list_ave,
					np.column_stack(
						(
							np.mean(res['Z_valid_flat']), 
							np.mean(res['Z_gt_valid_flat'])
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
		min_depth = draw_list_ave[:,1].min()
		max_depth = draw_list_ave[:,1].max()
		plt.plot([min_depth, max_depth], [min_depth, max_depth])	
		plt.plot(draw_list_ave[:,1],draw_list_ave[:,0],'ro')
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.draw()
		return
