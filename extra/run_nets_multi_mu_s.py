import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from utils import *
from skimage import data as skidata
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_laplacian

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "focalFlowNet_multi_mu_s"
method = "experiment"
dataset = [
	"test_1",
	"test_2",
	"test_3",
	"test_4",
	"test_5",
	"test_6",
	"test_7",
	"test_8",
	"test_9",
	"test_10",
	"test_11",
	"test_12",
	"test_13",
	"test_14",
	"test_15",
	"test_16",
	"test_17",
	# "train_1",
	# "train_2",
	# "train_3",
	# "train_4",
	# "train_5",
	# "train_6",
	# "train_7",
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
		cfg = data['cfg']
		mu_s = data['mu_s']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)
		mu_s = np.concatenate((mu_s,data['mu_s']),axis=0)

# cross out some of the data to speed up
I = I[:,189:411,369:591,:]

###########################################################
##  Determine the initial configuration
###########################################################
######## READ FROM FILES ########################
cfg_file = "./opt_results/"+netName+"/train.pickle"
with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']

######## CONVOLUTIONAL WINDOW ######################
cfg['szx_sensor'] = int(I.shape[2])
cfg['szy_sensor'] = int(I.shape[1])

cfg['valid_patch_x'] = 2
cfg['valid_patch_y'] = cfg['valid_patch_x']

# cfg['len_wx'] = \
# 	cfg['szx_sensor']-cfg['valid_patch_x']-(cfg['fx'].shape[1]-1)*2+1
# cfg['len_wy'] = \
# 	cfg['szy_sensor']-cfg['valid_patch_y']-(cfg['fy'].shape[0]-1)*2+1
cfg['len_wx'] = 201
cfg['len_wy'] = 201
cfg['wx'] = np.ones([1, cfg['len_wx']])
cfg['wy'] = np.ones([cfg['len_wy'], 1])
cfg['w'] = np.ones([cfg['len_wy'], cfg['len_wx']])

######## ASSERTION BEFORE RUNNING ##################
# if the size of the image is odd number
# the size of the valid patch should also be odd
assert(np.mod(cfg['szx_sensor']-cfg['valid_patch_x'],2)==0)
assert(np.mod(cfg['szy_sensor']-cfg['valid_patch_y'],2)==0)


############################################################

# range of output
depth = np.reshape(Loc[:,2,:]-cfg['Z_0'],-1)
DEPTH_RANGE = [depth.min(),depth.max()]
KEY_RANGE['Z'] = DEPTH_RANGE
KEY_RANGE['Z_gt'] = DEPTH_RANGE
KEY_RANGE['Z_err'] = [0,0.03]
KEY_RANGE['conf'] = [0,1]
KEY_RANGE['I1'] = [0,255]
KEY_RANGE['I2'] = [0,255]

# just some basic tryouts
ff = eval(netName+"(cfg)")
draw_list_ave = np.empty((0,2),dtype = np.float32)
draw_list_std = np.empty((0,2),dtype = np.float32)
for i in range(I.shape[0]):
	Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
	Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
	if cfg['noise_var'] > 0:
		I_input = gauss_noise(I=I[i,:,:,:], mu=0,var=cfg['noise_var'])
	else:
		I_input = I[i,:,:,:]
	ff.input_images(I_input, Z_map_gt, mu_s[i])
	
	# ff.regular_output()
	# cv2.waitKey(1)

	# Query some results for drawing
	query_list = ['Z_valid_flat', 'Z_gt_valid_flat']
	res = ff.query_results(query_list)

	# Update the derivative
	res['Z_valid_flat'] = res['Z_valid_flat']
	res['Z_gt_valid_flat'] = res['Z_gt_valid_flat']
	mean_Z = np.mean(res['Z_valid_flat'])
	mean_Z_gt = np.mean(res['Z_gt_valid_flat'])
	draw_list_ave = np.concatenate(
		(draw_list_ave,
			np.column_stack((mean_Z_gt,mean_Z))
		),axis = 0
	)
	data_low = res['Z_valid_flat'][res['Z_valid_flat'] < mean_Z]
	data_high = res['Z_valid_flat'][res['Z_valid_flat']>= mean_Z]
	std_low = np.sqrt(np.mean((data_low - mean_Z)**2))
	std_high = np.sqrt(np.mean((data_high - mean_Z)**2))
	draw_list_std = np.concatenate(
		(draw_list_std,
			np.column_stack((std_low,std_high))
		),axis = 0
	)

# draw the result before training
min_depth = draw_list_ave[:,0].min()
max_depth = draw_list_ave[:,0].max()
plt.plot([min_depth, max_depth], [min_depth, max_depth])	
plt.errorbar(draw_list_ave[:,0],draw_list_ave[:,1],\
	 yerr=[draw_list_std[:,0],draw_list_std[:,1]],fmt='.')
plt.axis([min_depth, max_depth, min_depth, max_depth])
plt.ylabel('Estimated depth (m)')
plt.xlabel('True depth (m)')

plt.show()