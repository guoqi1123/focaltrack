import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from utils import *
import bp_filtering

#####################################################################
##  Write down the net we want to use
#####################################################################
netName = "lensFlowNet"
method = "experiment"
dataset = [
	# "0-0002g",
	"0-0003g",
	"0-0004g",
	# "3-0003g",
	# "3-0004g"
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
		I_full = data['I']
		Loc = data['Loc']
	else:
		I_full = np.concatenate((I_full,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)

######## READ FROM FILES ########################
cfg_file = "./opt_results/"+netName+"/1x1t-text12-setup0-filtered1-1"+".pickle"
with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']

for idx in cfg.keys():
	#####################################################################
	# filtering
	I = bp_filtering.bp_filter_batch(\
		I_full, cfg[idx]['image_f'][0], cfg[idx]['image_f'][1])

	###########################################################
	##  Determine the initial configuration
	###########################################################
	if netName == "focalFlowNet":
		######## CONVOLUTIONAL WINDOW ######################
		cfg[idx]['valid_patch_x'] = 2
		cfg[idx]['valid_patch_y'] = cfg[idx]['valid_patch_x']

		# cfg[idx]['separable'] = True
		cfg[idx]['len_wx'] = \
			cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x']\
			-(cfg[idx]['fx'].shape[1]-1)*2+1
		cfg[idx]['len_wy'] = \
			cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y']\
			-(cfg[idx]['fy'].shape[0]-1)*2+1
		cfg[idx]['wx'] = np.ones([1, cfg[idx]['len_wx']])
		cfg[idx]['wy'] = np.ones([cfg[idx]['len_wy'], 1])
		cfg[idx]['w'] = np.ones([cfg[idx]['len_wy'], cfg[idx]['len_wx']])

		######## OPTICAL PARAMETERS ########################
		# cfg[idx]['Sigma'] = 0.00029

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		assert(np.mod(cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x'],2)==0)
		assert(np.mod(cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y'],2)==0)

	if netName == "lensFlowNet":
		######## CONVOLUTIONAL WINDOW ######################
		cfg[idx]['valid_patch_x'] = 2
		cfg[idx]['valid_patch_y'] = cfg[idx]['valid_patch_x']

		cfg[idx]['separable'] = True
		cfg[idx]['len_wx'] = \
			cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x']\
			-cfg[idx]['lap'].shape[1]+2
		cfg[idx]['len_wy'] = \
			cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y']\
			-cfg[idx]['lap'].shape[0]+2
		cfg[idx]['wx'] = np.ones([1, cfg[idx]['len_wx']])
		cfg[idx]['wy'] = np.ones([cfg[idx]['len_wy'], 1])
		cfg[idx]['w'] = np.ones([cfg[idx]['len_wy'], cfg[idx]['len_wx']])

		# cfg[idx]['wx'] = np.ones([1,5])
		# cfg[idx]['wy'] = np.ones([5,1])
		# cfg[idx]['w'] = np.ones([5,5])

		# ######## OPTICAL PARAMETERS ########################
		# cfg[idx]['noise_var'] = 1e-5

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		# assert(np.mod(cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x'],2)==0)
		# assert(np.mod(cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y'],2)==0)

	if netName == "lensFlowNet1Df":
		######## CONVOLUTIONAL WINDOW ######################
		cfg[idx]['valid_patch_x'] = 2
		cfg[idx]['valid_patch_y'] = cfg[idx]['valid_patch_x']

		cfg[idx]['separable'] = True
		cfg[idx]['len_wx'] = \
			cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x']-(cfg[idx]['fx'].shape[1]-1)*2+1
		cfg[idx]['len_wy'] = \
			cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y']-(cfg[idx]['fy'].shape[0]-1)*2+1
		cfg[idx]['wx'] = np.ones([1, cfg[idx]['len_wx']])
		cfg[idx]['wy'] = np.ones([cfg[idx]['len_wy'], 1])
		cfg[idx]['w'] = np.ones([cfg[idx]['len_wy'], cfg[idx]['len_wx']])


		# ######## OPTICAL PARAMETERS ########################
		# cfg[idx]['noise_var'] = 1e-5

		######## ASSERTION BEFORE RUNNING ##################
		# if the size of the image is odd number
		# the size of the valid patch should also be odd
		assert(np.mod(cfg[idx]['szx_sensor']-cfg[idx]['valid_patch_x'],2)==0)
		assert(np.mod(cfg[idx]['szy_sensor']-cfg[idx]['valid_patch_y'],2)==0)

	############################################################

	# range of output
	depth = np.reshape(Loc[:,2,:]-cfg[idx]['Z_0'],-1)
	DEPTH_RANGE = [depth.min(),depth.max()]
	KEY_RANGE['Z'] = DEPTH_RANGE
	KEY_RANGE['Z_gt'] = DEPTH_RANGE
	KEY_RANGE['Z_err'] = [0,0.03]
	KEY_RANGE['I1'] = [0,255]
	KEY_RANGE['I2'] = [0,255]

	# just some basic tryouts
	ff = eval(netName+"(cfg[idx])")
	draw_list_ave = np.empty((0,2),dtype = np.float32)
	draw_list_std = np.empty((0,2),dtype = np.float32)
	for i in range(I.shape[0]):
		Z_gt = Loc[i,2,int((Loc.shape[2]-1)/2)]
		Z_map_gt = np.ones(I.shape[1:3]) * Z_gt
		if cfg[idx]['noise_var'] > 0:
			I_input = gauss_noise(I=I[i,:,:,:], mu=0,var=cfg[idx]['noise_var'])
		else:
			I_input = I[i,:,:,:]
		ff.input_images(I_input, Z_map_gt)
		# ff.regular_output()
		# cv2.waitKey(1)
		# Query some results for drawing
		query_list = ['Z_valid_flat', 'Z_gt_valid_flat']
		res = ff.query_results(query_list)
		# Update the derivative
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
	# plt.plot([min_depth, max_depth], [min_depth, max_depth])	
	plt.errorbar(draw_list_ave[:,0],draw_list_ave[:,1],\
		 yerr=[draw_list_std[:,0],draw_list_std[:,1]],fmt='ro')
	# plt.axis([min_depth, max_depth, min_depth, max_depth])
	plt.ylabel('Estimated depth (m)')
	plt.xlabel('True depth (m)')

	plt.show()