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
import bp_filtering

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "lensFlowNet"
method = "experiment"
dataset = [
	# "sin0.000025-r0.001-dP0.2",
	"0-0001g",
	"0-0002g",
	#"0002",
	#"0002g",
	#"0003",
	#"0003g",
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
		I_full = data['I']
		Loc = data['Loc']
		cfg_init = data['cfg']
	else:
		I_full = np.concatenate((I_full,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)

#####################################################################
# filtering
image_fs = [
	[0,0],
	[1,1],
	[2,2],
	[3,3],
	[4,4],
	[5,5],
	[6,6],
	[7,7],
	[8,8],
]

# add everything to the configuration
cfg_init['netName'] = netName
cfg_init['dataName'] = dataset
cfg_init['total_num'] = I_full.shape[0]

# range of output
depth = np.reshape(Loc[:,2,:]-cfg_init['Z_0'],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# find the file to save
os.chdir('./opt_results/'+cfg_init['netName']+'_f'+'/')
lpickle = len(glob.glob('*.pickle'))
fileName = os.path.join(\
	str(lpickle)+".pickle"
)

cfg = {}
for image_f in image_fs:
	f_idx = str(image_f)
	cfg[f_idx] = cfg_init
	I = bp_filtering.bp_filter_batch(\
		I_full, image_f[0], image_f[1])
	print("Current image frequency: ", image_f)
	cfg[f_idx]['image_f'] = image_f

	#####################################################################
	##  Determine the initial configuration
	#####################################################################
	# just some basic tryouts
	if netName == "focalFlowNet":
		######## DIFFERENTIAL FILTERS #####################
		cfg[f_idx]['fx'] = np.array([[0,0,0,0,0.5,0,-0.5,0,0,0,0]])
		# cfg[f_idx]['fx'] = np.array(
		# 	[[0.5169,0.4626,0.4308,0.4325,0.2357,0,
		# 	-0.2357,-0.4325,-0.4308,-0.4626,-0.5169]]
		# )
		cfg[f_idx]['fy'] = np.transpose(cfg[f_idx]['fx'])

		cfg[f_idx]['fxx'] = signal.convolve2d(
				cfg[f_idx]['fx'],cfg[f_idx]['fx'],mode='full'
			)
		cfg[f_idx]['fyy'] = signal.convolve2d(
				cfg[f_idx]['fy'],cfg[f_idx]['fy'],mode='full'
			)
		cfg[f_idx]['fxy'] = signal.convolve2d(
				cfg[f_idx]['fx'],cfg[f_idx]['fy'],mode='full'
			)
		cfg[f_idx]['fyx'] = signal.convolve2d(
				cfg[f_idx]['fy'],cfg[f_idx]['fx'],mode='full'
			)

		######## CONVOLUTIONAL WINDOW ######################
		cfg[f_idx]['valid_patch_x'] = 2
		cfg[f_idx]['valid_patch_y'] = cfg[f_idx]['valid_patch_x']

		cfg[f_idx]['separable'] = True
		cfg[f_idx]['len_wx'] = \
			cfg[f_idx]['szx_sensor']-cfg[f_idx]['valid_patch_x']\
			-(cfg[f_idx]['fx'].shape[1]-1)*2+1
		cfg[f_idx]['len_wy'] = \
			cfg[f_idx]['szy_sensor']-cfg[f_idx]['valid_patch_y']\
			-(cfg[f_idx]['fy'].shape[0]-1)*2+1
		cfg[f_idx]['wx'] = np.ones([1, cfg[f_idx]['len_wx']])
		cfg[f_idx]['wy'] = np.ones([cfg[f_idx]['len_wy'], 1])
		cfg[f_idx]['w'] = np.ones([cfg[f_idx]['len_wy'], \
			cfg[f_idx]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		cfg[f_idx]['Sigma'] = 0.001
		cfg[f_idx]['dSigma_ratio'] = 0.000001 #Decrease dSigma to make it stable
		
		cfg[f_idx]['Z_0'] = 0

		cfg[f_idx]['noise_var'] = 1e-5

		######## TRAINING STUFFS ###########################
		cfg[f_idx]['learn_rate'] = 0.1 	# only for SGD
		cfg[f_idx]['step'] = 1 			# only for brute-force - better to use sparse loss
		cfg[f_idx]['step_thre'] = 0.05		# only for brute-force
		cfg[f_idx]['max_iter'] = 50 	
		cfg[f_idx]['batch_size'] = 150

		cfg[f_idx]['err_func'] = ['one_norm_err','sig_err']
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[f_idx]['train_range'] = [
			np.array([depth.min(),depth.max()]),
			np.array([depth.min(),depth.max()])
		]
		cfg[f_idx]['der_var'] = [
			['dLdfx','dLdfy', 'dLdSigma'],
			['dLdSigma']
		]

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		assert(np.mod(cfg[f_idx]['szx_sensor']\
			-cfg[f_idx]['valid_patch_x'],2)==0)
		assert(np.mod(cfg[f_idx]['szy_sensor']\
			-cfg[f_idx]['valid_patch_y'],2)==0)

	if netName == "lensFlowNet":
		# ######## READ FROM FILES ########################
		# cfg[f_idx]_file = "./opt_results/"+netName+"/1x1t-text3-setup2"+".pickle"
		# with open(cfg[f_idx]_file,'rb') as f:
		# 	cfg[f_idx]_data = pickle.load(f)
		# cfg[f_idx] = cfg[f_idx]_data['cfg[f_idx]']

		######### DIFFERENTIAL FILTERS #####################
		hf_len = 11
		cfg[f_idx]['lap'] = np.zeros((hf_len*2+1,hf_len*2+1))
		cfg[f_idx]['lap'][hf_len, hf_len] = -2
		cfg[f_idx]['lap'][hf_len-2,hf_len] = 0.5
		cfg[f_idx]['lap'][hf_len+2,hf_len] = 0.5
		cfg[f_idx]['lap'][hf_len,hf_len-2] = 0.5
		cfg[f_idx]['lap'][hf_len,hf_len+2] = 0.5

		cfg[f_idx]['fave'] = [[[0.5,0.5]]]
		
		hf_len_ft = 0
		cfg[f_idx]['ft'] = np.zeros((hf_len_ft*2+1,hf_len_ft*2+1,2))
		cfg[f_idx]['ft'][hf_len_ft,hf_len_ft,0] = -0.5
		cfg[f_idx]['ft'][hf_len_ft,hf_len_ft,1] = 0.5

		######## CONVOLUTIONAL WINDOW ######################
		# cfg[f_idx]['valid_patch_x'] = 2
		# cfg[f_idx]['valid_patch_y'] = cfg[f_idx]['valid_patch_x']

		# cfg[f_idx]['separable'] = True
		# cfg[f_idx]['len_wx'] = \
		# 	cfg[f_idx]['szx_sensor']-cfg[f_idx]['valid_patch_x']\
		# 	-cfg[f_idx]['lap'].shape[1]+2
		# cfg[f_idx]['len_wy'] = \
		# 	cfg[f_idx]['szy_sensor']-cfg[f_idx]['valid_patch_y']\
		# 	-cfg[f_idx]['lap'].shape[0]+2
		
		cfg[f_idx]['len_wx'] = 21
		cfg[f_idx]['len_wy'] = cfg[f_idx]['len_wx']
		cfg[f_idx]['wx'] = np.ones([1, cfg[f_idx]['len_wx']])
		cfg[f_idx]['wy'] = np.ones([cfg[f_idx]['len_wy'], 1])
		cfg[f_idx]['w'] = np.ones(\
			[cfg[f_idx]['len_wy'], cfg[f_idx]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		cfg[f_idx]['noise_var'] = 0
		cfg[f_idx]['a0'] = 4.
		cfg[f_idx]['da0_ratio'] = 1.
		cfg[f_idx]['a1'] = .8
		cfg[f_idx]['da1_ratio'] = 1.
		cfg[f_idx]['Z_0'] = -1.4
		cfg[f_idx]['dZ_0_ratio'] = 1.

		######## TRAINING STUFFS ###########################
		cfg[f_idx]['learn_rate'] = 0.1 	# only for SGD
		cfg[f_idx]['step'] = 0.1 			# only for brute-force - better to use sparse loss
		cfg[f_idx]['step_thre'] = 1e-5		# only for brute-force
		cfg[f_idx]['max_iter'] = 10 	
		cfg[f_idx]['err_func'] = ['half_norm_err']
		cfg[f_idx]['batch_size'] = 1000
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[f_idx]['train_range'] = [
			np.array([depth.min(),depth.max()]),
		]
		cfg[f_idx]['der_var'] = {
			'half_norm_err': ['dLdlap', 'dLda0','dLda1','dLdZ_0'],
		}
		

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		# assert(np.mod(cfg[f_idx]['szx_sensor']-cfg[f_idx]['valid_patch_x'],2)==0)
		# assert(np.mod(cfg[f_idx]['szy_sensor']-cfg[f_idx]['valid_patch_y'],2)==0)

	if netName == "lensFlowNet1Df":
		# ######## READ FROM FILES ########################
		# cfg[f_idx]_file = "./opt_results/"+netName+"/0"+".pickle"
		# with open(cfg[f_idx]_file,'rb') as f:
		# 	cfg[f_idx]_data = pickle.load(f)
		# cfg[f_idx] = cfg[f_idx]_data['cfg[f_idx]']

		######### DIFFERENTIAL FILTERS #####################
		# cfg[f_idx]['fx'] = np.array([[0.5,0,-0.5]])
		cfg[f_idx]['fx'] = np.array([[0,0,0,0,0.5,0,-0.5,0,0,0,0]])
		# cfg[f_idx]['fx'] = np.array(
		# 	[[0.5169,0.4626,0.4308,0.4325,0.2357,0,
		# 	-0.2357,-0.4325,-0.4308,-0.4626,-0.5169]]
		# )
		# randomly generated filter
		# hf = 0
		# hf_len = 11
		# while(np.sum(hf) < 0.5):
		# 	hf = np.random.rand(hf_len)
		# 	hf = hf/np.sum(np.abs(-hf))
		# cfg[f_idx]['fx'] = np.expand_dims(np.concatenate((hf,[0],-hf)),axis=0)
		
		cfg[f_idx]['fy'] = np.transpose(cfg[f_idx]['fx'])

		cfg[f_idx]['fxx'] = signal.convolve2d(
				cfg[f_idx]['fx'],cfg[f_idx]['fx'],mode='full'
			)
		cfg[f_idx]['fyy'] = signal.convolve2d(
				cfg[f_idx]['fy'],cfg[f_idx]['fy'],mode='full'
			)
		cfg[f_idx]['fxy'] = signal.convolve2d(
				cfg[f_idx]['fx'],cfg[f_idx]['fy'],mode='full'
			)
		cfg[f_idx]['fyx'] = signal.convolve2d(
				cfg[f_idx]['fy'],cfg[f_idx]['fx'],mode='full'
			)

		cfg[f_idx]['ft'] = np.zeros((cfg[f_idx]['fyy'].shape[0],\
			cfg[f_idx]['fxx'].shape[1],2))
		cfg[f_idx]['ft'][(cfg[f_idx]['fyy'].shape[0]-1)/2,\
			(cfg[f_idx]['fxx'].shape[1]-1)/2,0] = -0.5
		cfg[f_idx]['ft'][(cfg[f_idx]['fyy'].shape[0]-1)/2,\
			(cfg[f_idx]['fxx'].shape[1]-1)/2,1] = 0.5

		######## CONVOLUTIONAL WINDOW ######################
		cfg[f_idx]['valid_patch_x'] = 2
		cfg[f_idx]['valid_patch_y'] = cfg[f_idx]['valid_patch_x']

		cfg[f_idx]['separable'] = True
		cfg[f_idx]['len_wx'] = \
			cfg[f_idx]['szx_sensor']-cfg[f_idx]['valid_patch_x']\
			-(cfg[f_idx]['fx'].shape[1]-1)*2+1
		cfg[f_idx]['len_wy'] = \
			cfg[f_idx]['szy_sensor']-cfg[f_idx]['valid_patch_y']\
			-(cfg[f_idx]['fy'].shape[0]-1)*2+1
		cfg[f_idx]['wx'] = np.ones([1, cfg[f_idx]['len_wx']])
		cfg[f_idx]['wy'] = np.ones([cfg[f_idx]['len_wy'], 1])
		cfg[f_idx]['w'] = np.ones(\
			[cfg[f_idx]['len_wy'], cfg[f_idx]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		cfg[f_idx]['noise_var'] = 0
		cfg[f_idx]['a0'] = 1.
		cfg[f_idx]['da0_ratio'] = 1.
		cfg[f_idx]['a1'] = 1.
		cfg[f_idx]['da1_ratio'] = 1.
		cfg[f_idx]['Z_0'] = -1.4
		cfg[f_idx]['dZ_0_ratio'] = 1.

		######## TRAINING STUFFS ###########################
		cfg[f_idx]['learn_rate'] = 0.1 	# only for SGD
		cfg[f_idx]['step'] = 0.1 			# only for brute-force - better to use sparse loss
		cfg[f_idx]['step_thre'] = 1e-5		# only for brute-force
		cfg[f_idx]['max_iter'] = 50 	
		cfg[f_idx]['err_func'] = ['half_norm_err','sig_err']
		cfg[f_idx]['batch_size'] = 1000
		# use a shorter range to train
		depth = np.reshape(Loc[:,2,:],-1)
		cfg[f_idx]['train_range'] = [
			np.array([depth.min(),depth.max()]),
			np.array([-0.5,-0.7])
		]
		cfg[f_idx]['der_var'] = {
				'half_norm_err': ['dLdfxx','dLdfyy','dLda0','dLda1','dLdZ_0'],
				'sig_err': ['dLda0']
			}
		

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		assert(np.mod(cfg[f_idx]['szx_sensor']-cfg[f_idx]['valid_patch_x'],2)==0)
		assert(np.mod(cfg[f_idx]['szy_sensor']-cfg[f_idx]['valid_patch_y'],2)==0)

	#####################################################################

	# initailization
	ff = eval("training_"+netName+"(cfg[f_idx])")

	# adding noise to the data
	if cfg[f_idx]['noise_var'] > 0:
		for i in range(I.shape[0]):
			I[i,:,:,:] = gauss_noise(I[i,:,:,:],\
				 mu=0,var=cfg[f_idx]['noise_var'])

	# brute force training
	while(ff.cur_err_idx < len(ff.cfg['err_func'])):
		num_epi = 0
		step = cfg[f_idx]['step']
		step_thre = cfg[f_idx]['step_thre']
		max_iter = cfg[f_idx]['max_iter']
		print("Current error function is: "+\
			ff.cfg['err_func'][ff.cur_err_idx])
		
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
				step = ff.one_step_training_force(\
					I_temp, Loc_temp, step, step_thre)
			else:
				step = ff.one_step_training_force(\
					I_cur, Loc_cur, step, step_thre)

			num_epi += 1

			# update the values according to the training
			for key in cfg[f_idx].keys():
				if key in ff.vars.keys():
					cfg[f_idx][key] = ff.session.run(ff.vars[key])

			# save the result each time so its easier to visualize
			with open(fileName,'wb') as f:
				cfg_data = {
					'cfg':		cfg,
				}
				# dump the data into the file
				pickle.dump(cfg_data, f)

		ff.cur_err_idx += 1
		ff.cur_err_func = ff.cfg['err_func'][ff.cur_err_idx]
# show the training result finally
plt.show()


