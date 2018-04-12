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
netName = "pyConfLensFlowNetFast_offset"
method = "experiment"
#####################################################################

exec("from "+netName+" import KEY_RANGE")
exec("from "+netName+" import "+netName)

# import the data
folder = "./experiment_data/focaltrack_data_clement/"
imgs = glob.glob(folder+'*.png')

I = [[], []]
Loc = []
Z_f = []
for i in range(0,len(imgs),2):
	I0 = cv2.imread(imgs[i],cv2.IMREAD_GRAYSCALE)
	I0 = cv2.resize(I0, (480,300)).astype(np.float32)
	I1 = cv2.imread(imgs[i+1],cv2.IMREAD_GRAYSCALE)
	I1 = cv2.resize(I1, (480,300)).astype(np.float32)
	I[0].append(I0)
	I[1].append(I1)
	num = imgs[i].split('\\')[1][0:-4].split('-')
	Loc.append([[0,0],[0,0],[num[0],num[0]]])
	Z_f.append(num[1])
Loc = np.stack(Loc,0).astype(np.float32)
Z_f = np.stack(Z_f,0).astype(np.float32)
I = np.stack([np.stack(I[0],0),np.stack(I[1],0)],-1)

#####################################################################
# 

#####################################################################
##  Determine the initial configuration
#####################################################################
# just some basic tryouts

######## READ INITIAL FROM FILES ########################
# we use the training result of pyLensFlowNet as initialization
cfg_file = "./params/"+\
	"1x1t-text34-py4-setup5-one-sequential-regularize-nothreshold.pickle"
	# "initial_nothreshold.pickle"
	# "1x1t-text34-setup3-py5-sparse-w2r-stochastic.pickle"

with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']

for i in range(len(cfg)):
	cfg[i]['Z_0'] = 0

# compute training time
time_train = cfg_data['time']
total_time = 0
total_episode = 0
for i in range(len(time_train)):
	for j in range(len(time_train[i])):
		total_time += time_train[i][j]
		total_episode += 1

print('Total time: ', total_time)
print('Total episode', total_episode)

# adding noise to the data
if cfg[0]['noise_var'] > 0:
	for i in range(I.shape[0]):
		I[i,:,:,:] = gauss_noise(\
			I[i,:,:,:], mu=0,var=cfg['noise_var'])

# range of output
depth = np.reshape(Loc[:,2,:],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE
KEY_RANGE['Z_valid'] = DEPTH_RANGE
KEY_RANGE['Z_gt_valid'] = DEPTH_RANGE

# initailization
ff = eval(netName+"(cfg)")

# # find the file to save
# os.chdir('./opt_results/'+cfg[0]['netName']+'/')

pdb.set_trace()

# ff.visual_mean_var(I, Loc, Z_f, conf_thre=0)
ff.visual_err_conf_map(I, Loc, Z_f, log=True)
# ff.sparsification_map(I, Loc, Z_f)
# ff.AUC_map(I, Loc, Z_f)
# pdb.set_trace()

start_time = time.time()
for i in range(I.shape[0]):
	# input images
	Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
	Z_map_gt = np.ones(I[i,:,:,0].shape) * Z_gt
	ff.input_images(I[i,:,:,:], Z_map_gt, Z_f[i])


	# show the depth map
	ff.regular_output(conf_thre=0.99)

	# validate the image pyramid
	# ff.validate_pyramid()

	cv2.waitKey(1)
print((I.shape[0])/(time.time()-start_time))