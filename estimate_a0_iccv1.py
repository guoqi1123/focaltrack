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
netName = "pyConfLensFlowNetFast_iccv1"
method = "experiment"
training = "pretraining"
dataset = [
	"1",
	"2",
	"3",
	# "4",
	# "5",
	# "6",
	# "7",
	# "8",
]
#####################################################################


exec("from training_"+netName+" import KEY_RANGE")
exec("from training_"+netName+" import training_"+netName)

# # import the data
for i in range(len(dataset)):
	fileName = "./"+method+"_data"+"/pyConfLensFlowNetFast/"\
		+dataset[i]+".pickle"
	with open(fileName,'rb') as f:
		data = pickle.load(f)

	if i == 0:
		I = data['I']
		Loc = data['Loc']
		cfg_init = data['cfg']
		offsets = data['offsets']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)
		offsets = np.concatenate((offsets, data['offsets']),axis=0)

# change the inch to m
# Loc = -Loc * 25.4
# only select certain offsets
offsets_range = [k*1000 for k in range(55,20,-1)]
offsets_range = [40000]
flgs = np.where([offsets[i] in offsets_range for i in range(len(offsets))])[0]
I = np.stack([I[flg,:,:,:] for flg in flgs],0)
offsets = np.array([offsets[flg] for flg in flgs])
offsets = np.reshape(offsets, [-1, len(offsets_range)])
Loc = np.stack([Loc[flg,:,:] for flg in flgs],0)
Loc = Loc[0::len(offsets_range),:,:]

# crop the image
I = I[:,86:300-86,176:480-176,:] # 128x128

# reshape the image sequence
I = np.transpose(np.reshape(I,[len(offsets_range),-1,I.shape[1],I.shape[2],2]),[1,2,3,4,0])
I = np.reshape(I,[-1,I.shape[1],I.shape[2],len(offsets_range)*2])


#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
if training == "pretraining":
	# create image pyramid as a list of tuples
	I_py, I_lap_py = create_pyramid_tuple(
		I[0:1,:,:,:], 
		layer_num = 4
	)

	cfg = []
	for i in range(len(I_py[0])):
		cfg.append(copy.deepcopy(cfg_init[i]))
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
		cfg[i]['b0'] = 1.13e-4
		cfg[i]['b1'] = -3.2
		cfg[i]['da0a1_ratio'] = 1e-0
		cfg[i]['db0_ratio'] = 1e-6
		cfg[i]['db1_ratio'] = 1e-1
		cfg[i]['Z_0'] = -1.39
		cfg[i]['dZ_0_ratio'] = 1e-0
		cfg[i]['offsets'] = offsets_range

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

		######## TRAINING STUFFS ###########################
		cfg[i]['step'] = 1e-4		# only for brute-force - better to use sparse loss
		cfg[i]['step_thre'] = 1e-9		# only for brute-force
		cfg[i]['max_iter'] = [50,30,30,30] 	
		cfg[i]['err_func'] = [\
			'softmax_err',
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
				'softmax_err': ['dLda0a1','dLdZ_0'],
			}

		#####################################################################
		# add everything to the configuration
		cfg[i]['netName'] = netName
		cfg[i]['dataName'] = dataset
		cfg[i]['total_num'] = I.shape[0]
	# a0a1
	cfg[0]['a0a1'] = 1.5
	cfg[1]['a0a1'] = 1
	cfg[2]['a0a1'] = 0.3
	cfg[3]['a0a1'] = 0.15


# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval("training_"+netName+"(cfg)")
			

A = []
b = []
Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
ff.input_images(I[i,:,:,:], Z_map_gt,offsets[i,:])
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
	ff.input_images(I[i,:,:,:], Z_map_gt, offsets[i,:])

	query_list_layered = [\
		'u_1','u_2'
	]
	res_layered = ff.query_results_layered(query_list_layered)

	a1 = 1/(1e-4*offsets_range[0]-3.5)
	a0a1 = np.array([6, 2.5, 1.3, 0.3]) * 2
	a0 = a0a1/a1
	for j in range(len(res_layered)):
		A[j][i,:] = (ff.cfg[2]['a1'] - (Z_gt-ff.cfg[j]['Z_0'])) * np.reshape(res_layered[j]['u_1'],[-1])
		b[j][i,:] = - (Z_gt-ff.cfg[j]['Z_0']) * np.reshape(res_layered[j]['u_2'],[-1])

for i in range(len(ff.cfg)):
	A[i] = np.reshape(A[i], [-1,1])
	b[i] = np.reshape(b[i], [-1,1])


# a0 = [np.sum(A[i] * b[i]) / np.sum(A[i]* A[i])\
# 		for i in range(len(ff.cfg))]
# print(a0)

# a0 = [4,2.5,1.3,0.3]
# print(ff.cfg[0]['a1'])
# pdb.set_trace()
fig = plt.figure()
idx = [(np.random.rand(10000) * A[i].shape[0]).astype(np.int) for i in range(len(ff.cfg))] 
for i in range(len(ff.cfg)):
	ax = fig.add_subplot(2,3,i+1, title= "layer "+str(i))
	ax.scatter(A[i][idx[i]],b[i][idx[i]],s=0.01)
	ax.plot(A[i][idx[i]],a0[i]*A[i][idx[i]])
	ax.axis('equal')
plt.show()
