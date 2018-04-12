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
from utils import *

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "pyLensFlowNet"
method = "experiment"
dataset = [
	"4-0003g",
	"4-0004g",
]
#####################################################################


exec("from training_"+netName+" import KEY_RANGE")
exec("from training_"+netName+" import training_"+netName)

# # import the data
for i in range(len(dataset)):
	fileName = "./"+method+"_data"+"/"+netName+"/"+dataset[i]+".pickle"
	# fileName = "./experiment_data/"+netName+"/0.pickle"
	with open(fileName,'rb') as f:
		data = pickle.load(f)

	if i == 0:
		I = data['I']
		Loc = data['Loc']
		cfg_init = data['cfg']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)

# adding noise to the data
cfg_init['noise_var'] = 0
if cfg_init['noise_var'] > 0:
	for i in range(I.shape[0]):
		I[i,:,:,:] = gauss_noise(\
			I[i,:,:,:], mu=0,var=cfg['noise_var'])

# create image pyramid
I_py, I_lap_py = create_pyramid(I, layer_num =5)
#####################################################################


#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
if netName == "pyLensFlowNet":
	# configuration should be a 
	cfg = []
	for i in range(len(I_py)):
		cfg.append(copy.deepcopy(cfg_init))
		######### DIFFERENTIAL FILTERS #####################
		cfg[i]['fave'] = [[[-0.5,-0.5]]]
		
		hf_len_ft = 0
		cfg[i]['ft'] = np.zeros((hf_len_ft*2+1,hf_len_ft*2+1,2))
		cfg[i]['ft'][hf_len_ft,hf_len_ft,0] = -0.5
		cfg[i]['ft'][hf_len_ft,hf_len_ft,1] = 0.5

		######## CONVOLUTIONAL WINDOW ######################
		cfg[i]['szx_sensor'] = I_py[i].shape[2]
		cfg[i]['szy_sensor'] = I_py[i].shape[1]

		cfg[i]['valid_patch_x'] = 2-np.mod(cfg[i]['szx_sensor'],2)
		cfg[i]['valid_patch_y'] = 2-np.mod(cfg[i]['szy_sensor'],2)

		cfg[i]['separable'] = True
		cfg[i]['len_wx'] = \
			cfg[i]['szx_sensor']-\
			cfg[i]['valid_patch_x']+1
		cfg[i]['len_wy'] = \
			cfg[i]['szy_sensor']-\
			cfg[i]['valid_patch_y']+1
		
		# cfg[i]['len_wx'] = 1
		# cfg[i]['len_wy'] = cfg[i]['len_wx']
		cfg[i]['wx'] = np.ones([1, cfg[i]['len_wx']])
		cfg[i]['wy'] = np.ones([cfg[i]['len_wy'], 1])
		cfg[i]['w'] = np.ones([cfg[i]['len_wy'], cfg[i]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		cfg[i]['a0'] = 1.
		cfg[i]['da0_ratio'] = 1e-0
		cfg[i]['a1'] = 1.
		cfg[i]['da1_ratio'] = 1e-0
		cfg[i]['Z_0'] = -1.380
		cfg[i]['dZ_0_ratio'] = 1e-2

		######## TRAINING STUFFS ###########################
		cfg[i]['learn_rate'] = 0.1 	# only for SGD
		cfg[i]['step'] = 0.1 			# only for brute-force - better to use sparse loss
		cfg[i]['step_thre'] = 1e-5		# only for brute-force
		cfg[i]['max_iter'] = [10,50,50] 	
		cfg[i]['err_func'] = ['half_norm_err','half_norm_err_1','sig_err']
		cfg[i]['batch_size'] = 1000
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[i]['train_range'] = [
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()]),
		]

		cfg[i]['wr'] = (\
			depth.min(),\
			depth.max(),\
		)

		# adjusting a_1, Z_0 will change the location and the slope 
		# of the curve, adjusting filter, a_0 (fix a_1) will change
		# the shape and slope of the curve,
		cfg[i]['der_var'] = {
			'half_norm_err': ['dLda1','dLdZ_0'],
			'half_norm_err_1': ['dLda0'],
			'sig_err': ['dLda0'],
		}
	

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		# assert(np.mod(cfg[i]['szx_sensor']-cfg[i]['valid_patch_x'],2)==0)
		# assert(np.mod(cfg[i]['szy_sensor']-cfg[i]['valid_patch_y'],2)==0)

		#####################################################################
		# add everything to the configuration
		cfg[i]['netName'] = netName
		cfg[i]['dataName'] = dataset
		cfg[i]['total_num'] = I.shape[0]

# manually set some other parameters
cfg[0]['a0'] = 77.10239
cfg[1]['a0'] = 83.2762
cfg[2]['a0'] = 72.1094
cfg[3]['a0'] = 57.52041
cfg[4]['a0'] = 84.7640

# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval("training_"+netName+"()")

# find the file to save
os.chdir('./opt_results/'+cfg[0]['netName']+'/')
lpickle = len(glob.glob('*.pickle'))
fileName = os.path.join(\
	str(lpickle)+".pickle"
)

# brute force training
for py_i in range(len(I_py)):
	ff.add_basicNet(cfg[py_i])
	while(ff.bNets[py_i].cur_err_idx < len(ff.bNets[py_i].cfg['err_func'])):
		ff.bNets[py_i].cur_err_func = \
			ff.bNets[py_i].cfg['err_func'][ff.bNets[py_i].cur_err_idx]
		num_epi = 0
		step = ff.bNets[py_i].cfg['step']
		step_thre = ff.bNets[py_i].cfg['step_thre']
		max_iter = ff.bNets[py_i].cfg['max_iter'][ff.bNets[py_i].cur_err_idx]
		print("Current error function is: "+\
			ff.bNets[py_i].cfg['err_func'][ff.bNets[py_i].cur_err_idx])
		
		# select out all images that is within the range
		idx = np.empty(0,dtype=np.int32)
		for i in range(ff.bNets[py_i].cfg['total_num']):
			Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
			# skip the images that is not in the range
			if Z_gt < ff.bNets[py_i].cfg['train_range'][ff.bNets[py_i].cur_err_idx].min() or \
				Z_gt > ff.bNets[py_i].cfg['train_range'][ff.bNets[py_i].cur_err_idx].max():
				continue
			idx = np.append(idx,[i], axis=0)
		I_py_cur = I_py[py_i][idx,:,:,:]
		I_lap_py_cur = I_lap_py[py_i][idx,:,:,:]
		Loc_cur = Loc[idx,:,:]	

		while(step  >  step_thre and num_epi < max_iter):
			# ff.bNets[py_i].one_step_training_SGD(I, Loc)
			print("Episode: ", num_epi)
			# random sampling to obtain a batch to train on 
			if Loc_cur.shape[0] > ff.bNets[py_i].cfg['batch_size']:
				idx = np.random.random_integers(
					0,Loc_cur.shape[0]-1,ff.bNets[py_i].cfg['batch_size']
				)
				I_temp = I_py_cur[idx,:,:,:]
				I_lap_temp = I_lap_py_cur[idx,:,:,:]
				Loc_temp = Loc_cur[idx,:,:]
				step = ff.bNets[py_i].one_step_training_force(\
					I_temp, \
					I_lap_temp, \
					Loc_temp, \
					step, \
					step_thre
				)
			else:
				# if the batch size is bigger than the actual
				# number of the image, use all
				step = ff.bNets[py_i].one_step_training_force(\
					I_py_cur, \
					I_lap_py_cur, \
					Loc_cur, \
					step, \
					step_thre
				)

			num_epi += 1

			# update the working range
			ff.bNets[py_i].find_wr(
				I_py_cur, \
				I_lap_py_cur, \
				Loc_cur, \
			)

			# update the values according to the training
			for key in cfg[py_i].keys():
				if key in ff.bNets[py_i].vars.keys():
					cfg[py_i][key] = ff.bNets[py_i].session.run(ff.bNets[py_i].vars[key])

			# save the result each time so its easier to visualize
			with open(fileName,'wb') as f:
				cfg_data = {
					'cfg':		cfg,
				}
				# dump the data into the file
				pickle.dump(cfg_data, f)

		ff.bNets[py_i].cur_err_idx += 1
	# show the training result finally
	plt.show()


