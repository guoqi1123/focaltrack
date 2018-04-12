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
import time

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "pyConfLensFlowNet"
method = "experiment"
dataset = [
	"4-0001g",
	# "4-0002g",
	# "4-0005g",
	# "4-0006g",
	# "4-0007g",
	# "4-0008g",
	# "4-0009g",
	# "4-0010g",
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
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)
#####################################################################


#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts
if netName == "pyConfLensFlowNet":
	######## READ INITIAL FROM FILES ########################
	# we use the training result of pyLensFlowNet as initialization
	cfg_file = "./opt_results/pyConfLensFlowNet/"+\
		"1x1t-text34-setup4-py5-w3r-sequential.pickle"
		# "1x1t-text34-setup3-py5-sparse-w2r-stochastic.pickle"

	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']

	# adding noise to the data
	if cfg[0]['noise_var'] > 0:
		for i in range(I.shape[0]):
			I[i,:,:,:] = gauss_noise(\
				I[i,:,:,:], mu=0,var=cfg['noise_var'])

	# create image pyramid as a list of tupless
	I_py, I_lap_py = create_pyramid_tuple(
		I, 
		layer_num = len(cfg)
	)


# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE

# initailization
ff = eval(netName+"(cfg)")

# # find the file to save
# os.chdir('./opt_results/'+cfg[0]['netName']+'/')


# ff.visual_heatmap(I_py, I_lap_py, Loc, conf_thre=-np.inf)
# ff.visual_err_conf_map(I_py, I_lap_py, Loc, log=True)
# ff.sparsification_map(I_py, I_lap_py, Loc)
# ff.AUC_map(I_py, I_lap_py, Loc)
# pdb.set_trace()

start_time = time.time()
for i in range(I.shape[0]):
	# input images
	Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
	Z_map_gt = np.ones(I_py[i][0].shape[0:2]) * Z_gt
	ff.input_images(I_py[i], I_lap_py[i], Z_map_gt)

	# show the depth map
	ff.regular_output(conf_thre=0.9)
	cv2.waitKey(1)
print((time.time()-start_time)/(I.shape[0]))