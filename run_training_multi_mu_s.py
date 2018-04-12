import tensorflow as tf
import numpy as np
import cv2
import pdb
import pickle
import matplotlib.pyplot as plt
import pickle
import json
import os, glob
from utils import *

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "focalFlowNet_multi_mu_s"
method = "experiment"
dataset = [
	"train_1",
	"train_2",
	"train_3",
	"train_4",
	"train_5",
	"train_6",
	"train_7",
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
		cfg = data['cfg']
		mu_s = data['mu_s']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc,data['Loc']),axis=0)
		mu_s = np.concatenate((mu_s,data['mu_s']),axis=0)

# cross out some of the data to speed up
I = I[:,189:411,369:591,:]

#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
######## DIFFERENTIAL FILTERS #####################
cfg['fx'] = np.array([[0,0,0,0,0.5,0,-0.5,0,0,0,0]])

cfg['fy'] = np.transpose(cfg['fx'])

cfg['fxx'] = signal.convolve2d(
		cfg['fx'],cfg['fx'],mode='full'
	)
cfg['fyy'] = signal.convolve2d(
		cfg['fy'],cfg['fy'],mode='full'
	)
cfg['fxy'] = signal.convolve2d(
		cfg['fx'],cfg['fy'],mode='full'
	)
cfg['fyx'] = signal.convolve2d(
		cfg['fy'],cfg['fx'],mode='full'
	)
cfg['ft'] = np.array([[[-0.5,0,0.5]]])

######## CONVOLUTIONAL WINDOW ######################
cfg['szx_sensor'] = int(I.shape[2])
cfg['szy_sensor'] = int(I.shape[1])

cfg['valid_patch_x'] = 2
cfg['valid_patch_y'] = cfg['valid_patch_x']

cfg['separable'] = True
# cfg['len_wx'] = \
# 	cfg['szx_sensor']-cfg['valid_patch_x']-(cfg['fx'].shape[1]-1)*2+1
# cfg['len_wy'] = \
# 	cfg['szy_sensor']-cfg['valid_patch_y']-(cfg['fy'].shape[0]-1)*2+1
cfg['len_wx'] = 201
cfg['len_wy'] = 201
cfg['wx'] = np.ones([1, cfg['len_wx']])
cfg['wy'] = np.ones([cfg['len_wy'], 1])
cfg['w'] = np.ones([cfg['len_wy'], cfg['len_wx']])

######## OPTICAL PARAMETERS ########################
cfg['Sigma'] = 0.00109
cfg['dSigma_ratio'] = 1e-6 #Decrease dSigma to make it stable

cfg['Z_0'] = cfg['Z_0']

cfg['noise_var'] = 0

######## TRAINING STUFFS ###########################
cfg['learn_rate'] = 0.1 	# only for SGD
cfg['step'] = 1 			# only for brute-force - better to use sparse loss
cfg['step_thre'] = 1e-3		# only for brute-force
cfg['max_iter'] = [50,50] 	
cfg['batch_size'] = 1000

cfg['err_func'] = [
	# 'one_norm_err',
	'one_norm_err1'
]
# use a shorter range to train
depth = np.reshape(Loc[:,2,:],-1)
cfg['train_range'] = [
	np.array([depth.min(),depth.max()]),
	np.array([depth.min(),depth.max()])
]
cfg['der_var'] = {
	# 'one_norm_err'		:	['dLdSigma'],
	'one_norm_err1'		:	['dLdfx','dLdfy','dLdSigma'],	
}

######## ASSERTION BEFORE RUNNING ##################
# if the size of the image is odd number
# the size of the valid patch should also be odd
assert(np.mod(cfg['szx_sensor']-cfg['valid_patch_x'],2)==0)
assert(np.mod(cfg['szy_sensor']-cfg['valid_patch_y'],2)==0)	

#####################################################################
# add everything to the configuration
cfg['netName'] = netName
cfg['dataName'] = dataset
cfg['total_num'] = I.shape[0]

# range of output
depth = np.reshape(Loc[:,2,:]-cfg['Z_0'],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval("training_"+netName+"(cfg)")

# adding noise to the data
if cfg['noise_var'] > 0:
	for i in range(I.shape[0]):
		I[i,:,:,:] = gauss_noise(I[i,:,:,:], mu=0,var=cfg['noise_var'])

# find the file to save
os.chdir('./opt_results/'+cfg['netName']+'/')
lpickle = len(glob.glob('*.pickle'))
fileName = os.path.join(\
	str(lpickle)+".pickle"
)

# brute force training
while(ff.cur_err_idx < len(ff.cfg['err_func'])):
	ff.cur_err_func = ff.cfg['err_func'][ff.cur_err_idx]
	num_epi = 0
	step = cfg['step']
	step_thre = cfg['step_thre']
	max_iter = cfg['max_iter'][ff.cur_err_idx]
	print("Current error function is: "+ff.cfg['err_func'][ff.cur_err_idx])
	
	# select out all images that is within the range
	idx = np.empty(0,dtype=np.int32)
	for i in range(ff.cfg['total_num']):
		Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
		# skip the images that is not in the range
		if Z_gt < ff.cfg['train_range'][ff.cur_err_idx].min() or \
			Z_gt > ff.cfg['train_range'][ff.cur_err_idx].max():
			continue
		idx = np.append(idx,[i], axis=0)

	I_cur = I[idx,:,:,:]
	Loc_cur = Loc[idx,:,:]
	mu_s_cur = mu_s[idx]	

	while(step  >  step_thre and num_epi < max_iter):
		# ff.one_step_training_SGD(I, Loc)
		print("Episode: ", num_epi)

		# random sampling to obtain a batch to train on 
		if Loc_cur.shape[0] > ff.cfg['batch_size']:
			idx = np.random.random_integers(
				0,Loc_cur.shape[0]-1,ff.cfg['batch_size']
			)
			I_temp = I_cur[idx,:,:,:]
			Loc_temp = Loc_cur[idx,:,:]
			mu_s_temp = mu_s_cur[idx]
			step = ff.one_step_training_force(I_temp, Loc_temp, mu_s_temp, step, step_thre)
		else:
			step = ff.one_step_training_force(I_cur, Loc_cur, mu_s_cur, step, step_thre)

		num_epi += 1

		# update the values according to the training
		for key in cfg.keys():
			if key in ff.vars.keys():
				cfg[key] = ff.session.run(ff.vars[key])

		# save the result each time so its easier to visualize
		with open(fileName,'wb') as f:
			cfg_data = {
				'cfg':		cfg,
			}
			# dump the data into the file
			pickle.dump(cfg_data, f)

	ff.cur_err_idx += 1
	# show the training result finally
	plt.show()


