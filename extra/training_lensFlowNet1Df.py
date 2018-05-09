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
class training_lensFlowNet1Df(training_focalFlowNet):
	"""constructor: initializes FocalFlowProcessor"""
	def __init__(self, cfg = {}):
		#default configuraiton
		self.cfg = {
			# finite difference coefficient
			'fx': np.array([[0.5,0,-0.5]]),
			'fy': np.array([[0.5],[0],[-0.5]]),
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
			'fxx':'dLdfxx',
			'fyy':'dLdfyy',
			'fave':'dLdfave',
			'ft':'dLdft',
			'a0':'dLda0',
			'a1':'dLda1',
			'Z_0':'dLdZ_0',
		}
		self.var_dict = {
			'dLdfxx':'fxx',
			'dLdfyy':'fyy',
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

		self.image_to_show = ['Z','Z_gt']
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
			fxx = tf.Variable(self.cfg['fxx'], dtype = tf.float32)
			fyy = tf.Variable(self.cfg['fyy'], dtype = tf.float32)
			fxy = tf.Variable(self.cfg['fxy'], dtype = tf.float32)
			fyx = tf.Variable(self.cfg['fyx'], dtype = tf.float32)
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
			I_xx = dIdx_batch(I_trans, fxx)
			I_yy = dIdy_batch(I_trans, fyy)
			I_xy = dIdx_batch(I_trans, fxy)
			I_yx = dIdx_batch(I_trans, fyx)
			I_lap_batch = I_xx + I_yy
			# conduct averaging
			I_lap_batch_trans = tf.transpose(I_lap_batch, perm = [1,2,0])
			I_lap = dIdt(I_lap_batch_trans, fave)

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

			Z = a0 * a1 / (-U[:,:,0]+a0)
			
			#save references to required I/O
			self.vars['U'] = U[:,:,0]
			self.vars['I'] = I
			self.vars['Z'] = Z
			self.vars['Z_gt'] = Z_gt0
			self.vars['t/lap'] = I_t/(I_lap+0.000000001)

			self.valid_U()

			# compute error
			self.vars['loss_func'] = {}
			for ef in self.cfg['err_func']:
				self.vars['loss_func'][ef] = \
					eval('self.'+ef+'()')

			####################################################################
			# below are derivatives and optimization
			####################################################################
			self.der['dLdfxx'] = {}
			self.der['dLdfyy'] = {}
			self.vars['fxx'] = fxx
			self.vars['fyy'] = fyy

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

			for ef in self.cfg['err_func']:
				dLdfxx, dLdfyy, dLdft, dLdfave = \
					tf.gradients(self.vars['loss_func'][ef], [fxx, fyy, ft, fave])
				num_inst = tf.to_float(tf.shape(self.vars['loss_func'][ef])[0])
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
				dLdft = dLdft + tf.transpose(dLdft, perm = [1,0,2])
				dLdft = dLdft + tf.reverse(dLdft, dims = [True, True, False])
				dLdft = dLdft / 8. / num_inst
				self.der['dLdft'][ef] = dLdft
				

				# derivative w.r.t. to fave
				dLdfave = dLdfave / num_inst
				self.der['dLdfave'][ef] = dLdfave

				#### derivative w.r.t. optical parameters
				dLda0, dLda1, dLdZ_0 = tf.gradients(
					self.vars['loss_func'][ef], [a0, a1, Z_0]
				)
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
			self.vars['t'] = I_t
			self.vars['xx'] = I_xx
			self.vars['yy'] = I_yy
			self.vars['xy'] = I_xy
			self.vars['yx'] = I_yx

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
	def valid_U(self):
		# find out valid regions for U
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
		
		self.vars['U_valid'] = self.vars['U'][
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut,
		]
		self.vars['U_valid_flat'] = tf.reshape(
			self.vars['U_valid'], [-1]
		)
		return 

	def one_step_training_force(self, I, Loc, step, min_step):
		# perform one step training by finding the step size
		# that is guaranteed to decrease loss
		print("Below is a new optimization step")
		draw_list = np.empty((0,2),dtype = np.float32)
		for i in range(I.shape[0]):
			# input images
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
			self.input_images(I[i,:,:,:], Z_map_gt)

			# # show the depth map
			# self.regular_output()
			# cv2.waitKey(1)

			# Query some results for drawing
			query_list = ['Z_valid_flat','Z_gt_valid_flat','U_valid_flat']
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
		draw_list = np.empty((0,2),dtype = np.float32)
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
			draw_list = np.concatenate(
				(draw_list,
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
		min_depth = draw_list[:,1].min()
		max_depth = draw_list[:,1].max()
		plt.plot([min_depth, max_depth], [min_depth, max_depth])	
		plt.plot(draw_list[:,1],draw_list[:,0],'ro')
		plt.axis([min_depth, max_depth, min_depth, max_depth])
		plt.draw()
		return
