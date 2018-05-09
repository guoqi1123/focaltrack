import tensorflow as tf
import numpy as np
import cv2
import pdb
import pickle
import matplotlib.pyplot as plt
import pickle
import json
import os, glob
import copy
import time as tm
from utils import *
import scipy.misc

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "pyConfLensFlowNetFast_offset"
method = "experiment"
training = "final-tuning"
dataset = [
	"5-0003g",
	"5-0004g",
]
#####################################################################


exec("from training_"+netName+" import KEY_RANGE")
exec("from training_"+netName+" import training_"+netName)

# # import the data
for i in range(len(dataset)):
	fileName = "./"+method+"_data"+"/pyConfLensFlowNetFast/"+dataset[i]+".pickle"
	# fileName = "./experiment_data/"+netName+"/0.pickle"
	with open(fileName,'rb') as f:
		data = pickle.load(f)

	if i == 0:
		I = data['I']
		offset = data['offset']
		Loc = data['Loc']
		cfg_init = data['cfg']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		offset = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)
#####################################################################

#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
if training == "pretraining":
	# adding noise to the data
	cfg_init['noise_var'] = 0
	if cfg_init['noise_var'] > 0:
		for i in range(I.shape[0]):
			I[i,:,:,:] = gauss_noise(\
				I[i,:,:,:], mu=0,var=cfg['noise_var'])

	# create image pyramid as a list of tuples
	I_py, I_lap_py = create_pyramid_tuple(
		I[0:1,:,:,:], 
		layer_num = 4
	)

	cfg = []
	for i in range(len(I_py[0])):
		cfg.append(copy.deepcopy(cfg_init))
		######### DIFFERENTIAL FILTERS #####################
		cfg[i]['gauss'] = np.array([\
				[0.0000,0.0013,0.0040,0.0013,0.0000],\
				[0.0013,0.0377,0.1162,0.0377,0.0013],\
				[0.0040,0.1162,0.3579,0.1162,0.0040],\
				[0.0013,0.0377,0.1162,0.0377,0.0013],\
				[0.0000,0.0013,0.0040,0.0013,0.0000],\
				])
		cfg[i]['fave'] = [[[-0.5,-0.5]]]
		cfg[i]['ext_f'] = np.array([
			[[0,0,0],[0,1,0],[0,0,0]],
			[[0,0,0],[0.5,0,-0.5],[0,0,0]],
			[[0,0.5,0],[0,0,0],[0,-0.5,0]]
		])
		
		hf_len_ft = 0
		cfg[i]['ft'] = np.zeros((hf_len_ft*2+1,hf_len_ft*2+1,2))
		cfg[i]['ft'][hf_len_ft,hf_len_ft,0] = -0.5
		cfg[i]['ft'][hf_len_ft,hf_len_ft,1] = 0.5

		######## CONVOLUTIONAL WINDOW ######################
		cfg[i]['szx_sensor'] = I_py[0][i].shape[1]
		cfg[i]['szy_sensor'] = I_py[0][i].shape[0]

		cfg[i]['valid_patch_x'] = 2-np.mod(cfg[i]['szx_sensor'],2)
		cfg[i]['valid_patch_y'] = 2-np.mod(cfg[i]['szy_sensor'],2)

		cfg[i]['separable'] = True
		cfg[i]['len_wx'] = \
			cfg[i]['szx_sensor']-\
			cfg[i]['valid_patch_x']+1
		cfg[i]['len_wy'] = \
			cfg[i]['szy_sensor']-\
			cfg[i]['valid_patch_y']+1
		
		# cfg[i]['len_wx'] = int(500/2**i)
		# cfg[i]['len_wy'] = cfg[i]['len_wx']
		cfg[i]['wx'] = np.ones([1, cfg[i]['len_wx']])
		cfg[i]['wy'] = np.ones([cfg[i]['len_wy'], 1])
		cfg[i]['w'] = np.ones([cfg[i]['len_wy'], cfg[i]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		cfg[i]['noise_var'] = 0.
		cfg[i]['a0'] = 1.
		cfg[i]['da0_ratio'] = 1e-0
		cfg[i]['c1'] = .600
		cfg[i]['dc1_ratio'] = 1e-0
		cfg[i]['c2'] = 1.
		cfg[i]['dc2_ratio'] = 1e-0
		cfg[i]['Z_0'] = -1.380
		cfg[i]['dZ_0_ratio'] = 1e-0

		######## WEIGHTED BASELINE CONFIDENCE ##############
		cfg[i]['w_bc'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['c3'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['c4'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['c5'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		
		######## POSTERIOR CONFIDENCE ######################
		cfg[i]['C_e0'] = 1.
		cfg[i]['e_ratio'] = .5
		cfg[i]['ft_ssum'] = .5
		cfg[i]['n2_ssum'] = .375

		######## TRAINING STUFFS ###########################
		cfg[i]['step'] = 1e-4		# only for brute-force - better to use sparse loss
		cfg[i]['step_thre'] = 1e-9		# only for brute-force
		cfg[i]['max_iter'] = [50,30,30,30] 	
		cfg[i]['err_func'] = [\
			'one_norm_err',
			'one_norm_err1',
		]
		cfg[i]['conf_func'] = 'w3_baseline_conf'
		cfg[i]['batch_size'] = 1000
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[i]['train_range'] = [
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
		]
		cfg[i]['der_var'] = {
				'one_norm_err': ['dLdw_bc','dLdc3','dLdc4','dLdc5'],
				'one_norm_err1': ['dLda0','dLdc1','dLdc2','dLdZ_0']
			}

		#####################################################################
		# add everything to the configuration
		cfg[i]['netName'] = netName
		cfg[i]['dataName'] = dataset
		cfg[i]['total_num'] = I.shape[0]

	# cfg[0]['a0'] = 0.1495
	# cfg[0]['a0'] = 11.2572
	# cfg[1]['a0'] = 7.4988
	# cfg[2]['a0'] = 2.9344
	# cfg[3]['a0'] = 1.0777

	cfg[0]['a0'] = 28.5538
	cfg[1]['a0'] = 20.9358
	cfg[2]['a0'] = 9.09133
	cfg[3]['a0'] = 3.74751

	# cfg[0]['w_bc'] = [999999,999999,999999]
	# cfg[0]['w_bc1'] = [999999,999999,999999]
	# cfg[0]['w_bc2'] = [999999,999999,999999]



	# record the training result
	loss = []
	time = []


if training == "final-tuning":
	######## READ INITIAL FROM FILES ########################
	# we use the training result of pyLensFlowNet as initialization
	cfg_file = "./opt_results/pyConfLensFlowNetFast_ext/"+\
		"1x1t-text34-py4-setup5-one-sequential-regularize-nothreshold.pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']
	loss = cfg_data['loss']
	time = cfg_data['time']

	# adding noise to the data
	if cfg[0]['noise_var'] > 0:
		for i in range(I.shape[0]):
			I[i,:,:,:] = gauss_noise(\
				I[i,:,:,:], mu=0,var=cfg['noise_var'])

	# create image pyramid as a list of tuples
	I_py, I_lap_py = create_pyramid_tuple(
		I[0:1,:,:,:], 
		layer_num = 5
	)

	for i in range(len(cfg)):
		# ######## WEIGHTED BASELINE CONFIDENCE ##############
		# cfg[i]['w_bc'] = 1.
		# cfg[i]['w_bc1'] = 1.
		# cfg[i]['w_bc2'] = 1.
		# cfg[i]['lo'] = np.reshape(Loc[:,2,:],-1).min()
		# cfg[i]['hi'] = np.reshape(Loc[:,2,:],-1).max()

		# ######## POSTERIOR CONFIDENCE ######################
		# cfg[i]['C_e0'] = 1.
		# cfg[i]['e_ratio'] = .5
		# cfg[i]['ft_ssum'] = .5
		# cfg[i]['n2_ssum'] = .375

		######## TRAINING STUFFS ###########################
		cfg[i]['step'] = 1e-4			# only for brute-force - better to use sparse loss
		cfg[i]['step_thre'] = 1e-9		# only for brute-force
		cfg[i]['max_iter'] = [50,30] 	
		cfg[i]['err_func'] = [\
			'sparsification_err',
			'sparsification_err1',
		]

		cfg[i]['conf_func'] = 'w3_baseline_conf'
		# cfg[i]['conf_func'] = 'w3r_baseline_conf'
		cfg[i]['batch_size'] = 1000
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[i]['train_range'] = [
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
		]
		cfg[i]['der_var'] = {
				'sparsification_err': ['dLdw_bc','dLdw_bc1','dLdw_bc2'],
				'sparsification_err1': ['dLda1','dLdZ_0','dLda0'],
			}

# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval("training_"+netName+"(cfg)")

# find the file to save
os.chdir('./opt_results/'+cfg[0]['netName']+'/')
lpickle = len(glob.glob('*.pickle'))
fileName = os.path.join(\
	str(lpickle)+".pickle"
)
# save the result each time so its easier to visualize
# with open(fileName,'wb') as f:
# 	cfg_data = {
# 		'cfg':		cfg,
# 		'loss':		loss,
# 		'time':		time,
# 	}
# 	# dump the data into the file
# 	pickle.dump(cfg_data, f)

# pdb.set_trace()

# brute force training
while(ff.cur_err_idx < len(ff.cfg[0]['err_func'])):
	loss.append([])
	time.append([])
	# set the current error function and opt parameters
	ff.cur_err_func = ff.cfg[0]['err_func'][ff.cur_err_idx]
	num_epi = 0
	step = cfg[0]['step']
	step_thre = cfg[0]['step_thre']
	max_iter = cfg[0]['max_iter'][ff.cur_err_idx]
	print("Current error function is: "+\
		ff.cfg[0]['err_func'][ff.cur_err_idx]
	)	

	if ff.cfg[0]['err_func'][ff.cur_err_idx][0:-1] == 'sparsification_err' or \
		ff.cfg[0]['err_func'][ff.cur_err_idx] == 'sparsification_err':
		temperature = 0.
	else:
		temperature = 0.

	while(step  >  step_thre and num_epi < max_iter):
		start_time = tm.time()
		# ff.one_step_training_SGD(I, Loc)
		print("Episode: ", num_epi)
		print("Temperature:", temperature)

		step, loss_tmp = ff.one_step_training_force(
			I, 
			Loc, 
			step, 
			step_thre,
			temperature,
		)

		loss[ff.cur_err_idx].append(loss_tmp)
		time[ff.cur_err_idx].append(tm.time() - start_time)

		num_epi += 1
		temperature /= 2

		# update the values according to the training
		for i in range(len(cfg)):
			for key in cfg[i].keys():
				if key in ff.vars[i].keys():
					cfg[i][key] = ff.session.run(ff.vars[i][key])

		# save the result each time so its easier to visualize
		with open(fileName,'wb') as f:
			cfg_data = {
				'cfg':		cfg,
				'loss':		loss,
				'time':		time,
			}
			# dump the data into the file
			pickle.dump(cfg_data, f)

	ff.cur_err_idx += 1
	# show the training result finally
	plt.show()
