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
	"3-0003g",
	# "3-0002g",
]
#####################################################################


exec("from "+netName+" import KEY_RANGE")
exec("from "+netName+" import "+netName)

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

#####################################################################


#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
if netName == "pyLensFlowNet":
	######## READ FROM FILES ########################
	cfg_file = "./opt_results/"+netName+"/1x1t-text34-setup3-py5-2"+".pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']

# create image pyramid
I_py, I_lap_py = create_pyramid(I, len(cfg))

# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval(netName+"()")

# brute force training
for py_i in range(len(I_py)):
	ff.add_basicNet(cfg[py_i])

# # draw the mean and variance 
# draw_list_ave = np.empty((I.shape[0],2),dtype = np.float32)
# draw_list_std = np.empty((I.shape[0],2),dtype = np.float32)

# used to draw the histogram
Z_flat = np.empty((0,), dtype = np.float32)
Z_gt_flat = np.empty((0,), dtype = np.float32)

Z_flat_layered = [
	np.empty((0,), dtype = np.float32)\
	for j in range(len(cfg))
]
Z_gt_flat_layered = [
	np.empty((0,), dtype = np.float32)\
	for j in range(len(cfg))
]

Zw_flat_layered = [
	np.empty((0,), dtype = np.float32)\
	for j in range(len(cfg))
]
Z_gtw_flat_layered = [
	np.empty((0,), dtype = np.float32)\
	for j in range(len(cfg))
]

for i in range(I.shape[0]):
	I_tmp = []
	I_lap_tmp = []
	for py_i in range(len(I_py)):
		I_tmp.append(I_py[py_i][i,:,:,:])
		I_lap_tmp.append(I_lap_py[py_i][i,:,:,:])

	# create the ground truth depth map
	Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
	Z_map_gt = np.ones(I_py[0].shape[1:3]) * Z_gt
	
	# feed in the data
	Z, Z_layered, Zw_layered = ff.compute_err(I_tmp, I_lap_tmp, Z_map_gt)
	
	# # record the statistics of the prediction
	# draw_list_ave[i,0] = Z_gt
	# draw_list_ave[i,1] = ff.mean_Z
	# draw_list_std[i,0] = ff.std_low
	# draw_list_std[i,1] = ff.std_high

	# draw the histogram
	Z_flat = np.concatenate(
		(Z_flat, Z.flatten()), axis=0
	)
	Z_gt_flat = np.concatenate(
		(Z_gt_flat, Z_map_gt.flatten()), axis=0
	)

	for j in range(len(cfg)):
		Z_flat_layered[j] = np.concatenate(
			(Z_flat_layered[j], Z_layered[j].flatten()), 
			axis=0
		)
		Z_map_gt = np.ones(I_py[j][i].shape[0:2]) * Z_gt
		Z_gt_flat_layered[j] = np.concatenate(
			(Z_gt_flat_layered[j], Z_map_gt.flatten()), 
			axis=0
		)

		Zw_flat_layered[j] = np.concatenate(
			(Zw_flat_layered[j], Zw_layered[j].flatten()), 
			axis=0
		)
		Z_map_gtw = np.ones(Zw_layered[j].shape) * Z_gt
		Z_gtw_flat_layered[j] = np.concatenate(
			(Z_gtw_flat_layered[j], Z_map_gtw.flatten()), 
			axis=0
		)

# # draw the mean and variance figure
# min_depth = draw_list_ave[:,0].min()
# max_depth = draw_list_ave[:,0].max()
# plt.plot([min_depth, max_depth], [min_depth, max_depth])	
# plt.errorbar(draw_list_ave[:,0],draw_list_ave[:,1],\
# 	 yerr=[draw_list_std[:,0],draw_list_std[:,1]],fmt='ro')
# plt.axis([min_depth, max_depth, min_depth, max_depth])
# plt.ylabel('Estimated depth (m)')
# plt.xlabel('True depth (m)')

# draw the histograms
fig = plt.figure()
ff.heatmap(\
	Z_flat, 
	Z_gt_flat, 
	I.shape[0], 
	fig, 
	3,4,1, 
	'fused_result'
)
ff.heatmap(\
	Z_flat_layered[0], 
	Z_gt_flat_layered[0], 
	I.shape[0], 
	fig, 
	3,4,2, 
	'layer 0'
)
ff.heatmap(\
	Z_flat_layered[1], 
	Z_gt_flat_layered[1], 
	I.shape[0], 
	fig, 
	3,4,3, 
	'layer 1'
)
ff.heatmap(\
	Z_flat_layered[2], 
	Z_gt_flat_layered[2], 
	I.shape[0], 
	fig, 
	3,4,4, 
	'layer 2'
)
ff.heatmap(\
	Z_flat_layered[3], 
	Z_gt_flat_layered[3], 
	I.shape[0], 
	fig, 
	3,4,5, 
	'layer 3'
)
ff.heatmap(\
	Z_flat_layered[4], 
	Z_gt_flat_layered[4], 
	I.shape[0], 
	fig, 
	3,4,6, 
	'layer 4'
)

# draw the histograms for windowed version
ff.heatmap(\
	Zw_flat_layered[0], 
	Z_gtw_flat_layered[0], 
	I.shape[0], 
	fig, 
	3,4,8, 
	'layer 0, windowed'
)
ff.heatmap(\
	Zw_flat_layered[1], 
	Z_gtw_flat_layered[1], 
	I.shape[0], 
	fig, 
	3,4,9, 
	'layer 1, windowed'
)
ff.heatmap(\
	Zw_flat_layered[2], 
	Z_gtw_flat_layered[2], 
	I.shape[0], 
	fig, 
	3,4,10, 
	'layer 2, windowed'
)
ff.heatmap(\
	Zw_flat_layered[3], 
	Z_gtw_flat_layered[3], 
	I.shape[0], 
	fig, 
	3,4,11, 
	'layer 3, windowed'
)
ff.heatmap(\
	Zw_flat_layered[4], 
	Z_gtw_flat_layered[4], 
	I.shape[0], 
	fig, 
	3,4,12, 
	'layer 4, windowed'
)

plt.show()