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

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "pyConfLensFlowNetFast_ext"
method = "experiment"
training = "pretraining"
dataset = [
	"5-0003g",
	"5-0004g",
]
#####################################################################


exec("from "+netName+" import KEY_RANGE")
exec("from "+netName+" import "+netName)


# # import the data
for i in range(len(dataset)):
	fileName = "./"+method+"_data"+"/pyConfLensFlowNetFast/"+dataset[i]+".pickle"
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
#####################################################################

# # resize the image
# I_new = np.empty((I.shape[0],int(I.shape[1]/2),int(I.shape[2]/2),I.shape[3]))
# for i in range(I_new.shape[0]):
# 	I_new[i,:,:,:] = tuple(pyramid_gaussian(\
# 		I[i,:,:,:], \
# 		max_layer=1, \
# 		downscale=2
# 	))[1]
# I = I_new


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
		cfg[i]['a1'] = .600
		cfg[i]['da1_ratio'] = 1e-2
		cfg[i]['Z_0'] = -1.380
		cfg[i]['dZ_0_ratio'] = 1e-2

		######## WEIGHTED BASELINE CONFIDENCE ##############
		cfg[i]['w_bc'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['w_bc1'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['w_bc2'] = np.array([1. for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['lo'] = np.array([np.reshape(Loc[:,2,:],-1).min() for i in range(cfg[i]['ext_f'].shape[0])])
		cfg[i]['hi'] = np.array([np.reshape(Loc[:,2,:],-1).max() for i in range(cfg[i]['ext_f'].shape[0])])

		######## POSTERIOR CONFIDENCE ######################
		cfg[i]['C_e0'] = 1.
		cfg[i]['e_ratio'] = .5
		cfg[i]['ft_ssum'] = .5
		cfg[i]['n2_ssum'] = .375

		cfg[i]['conf_func'] = 'w3r_baseline_conf'

		#####################################################################
		# add everything to the configuration
		cfg[i]['netName'] = netName
		cfg[i]['dataName'] = dataset
		cfg[i]['total_num'] = I.shape[0]

# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval(netName+"(cfg)")
			

A = []
b = []
Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
ff.input_images(I[i,:,:,:], Z_map_gt)
query_list_layered = [\
	'u_1','u_2'
]
res_layered = ff.query_results_layered(query_list_layered)

for i in range(len(ff.cfg)):
	rows = res_layered[i]['u_1'].shape[0]
	cols = res_layered[i]['u_1'].shape[1]
	A.append(
		np.empty(
			(
				I.shape[0],
				rows * cols
			)
		)
	)
	b.append(
		np.empty(
			(
				I.shape[0],
				rows * cols
			)
		)
	)

for i in range(I.shape[0]):
	# input images
	Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
	Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
	ff.input_images(I[i,:,:,:], Z_map_gt)

	query_list_layered = [\
		'u_1','u_2'
	]
	res_layered = ff.query_results_layered(query_list_layered)

	for j in range(len(res_layered)):
		A[j][i,:] = (ff.cfg[j]['a1'] - (Z_gt-ff.cfg[j]['Z_0'])) * np.reshape(res_layered[j]['u_1'],[-1])
		b[j][i,:] = - (Z_gt-ff.cfg[j]['Z_0']) * np.reshape(res_layered[j]['u_2'],[-1])

for i in range(len(ff.cfg)):
	A[i] = np.reshape(A[i], [-1,1])
	b[i] = np.reshape(b[i], [-1,1])


a0 = [np.sum(A[i] * b[i]) / np.sum(A[i]* A[i])\
		for i in range(len(ff.cfg))]
print(a0)

# a0 = [0.1747, 8.7591, 7.6753, 2.9596, 1.0967]
fig = plt.figure()
idx = [(np.random.rand(10000) * A[i].shape[0]).astype(np.int) for i in range(len(ff.cfg))] 
for i in range(len(ff.cfg)):
	ax = fig.add_subplot(2,3,i+1, title= "layer "+str(i))
	ax.scatter(A[i][idx[i]],b[i][idx[i]],s=0.01)
	ax.plot(A[i][idx[i]],a0[i]*A[i][idx[i]])
	ax.axis('equal')
plt.show()
