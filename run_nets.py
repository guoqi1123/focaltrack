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
netName = "lensFlowNet"
method = "experiment"
dataset = [
	# "0-0002g",
	"0-0002g",
	"0-0001g",
	# "0-0003g",
	# "0-0004g",
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
		I = data['I']
		Loc = data['Loc']
		cfg = data['cfg']
	else:
		I = np.concatenate((I,data['I']),axis=0)
		Loc = np.concatenate((Loc, data['Loc']),axis=0)


#####################################################################
# filtering
# I = I[:,300:600, 150:450,:]
# I = bp_filtering.bp_filter_batch(I, 0, 20)
# resizing
# I = bp_filtering.resize(I, 600*0.1, 960*0.1)

###########################################################
##  Determine the initial configuration
###########################################################
if netName == "focalFlowNet":
	######## READ FROM FILES ########################
	cfg_file = "./opt_results/"+netName+"/1"+".pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']

	# ######## DIFFERENTIAL FILTERS #####################
	# # cfg['fx'] = np.array([[0,0,0,0,0.5,0,-0.5,0,0,0,0]])
	# cfg['fx'] = np.array(
	# 	[[0.5169,0.4626,0.4308,0.4325,0.2357,0,
	# 	-0.2357,-0.4325,-0.4308,-0.4626,-0.5169]]
	# )
	# cfg['fy'] = np.transpose(cfg['fx'])

	# cfg['fxx'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fx'],mode='full'
	# 	)
	# cfg['fyy'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fy'],mode='full'
	# 	)
	# cfg['fxy'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fy'],mode='full'
	# 	)
	# cfg['fyx'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fx'],mode='full'
	# 	)

	######## CONVOLUTIONAL WINDOW ######################
	cfg['valid_patch_x'] = 2
	cfg['valid_patch_y'] = cfg['valid_patch_x']

	# cfg['separable'] = True
	cfg['len_wx'] = \
		cfg['szx_sensor']-cfg['valid_patch_x']-(cfg['fx'].shape[1]-1)*2+1
	cfg['len_wy'] = \
		cfg['szy_sensor']-cfg['valid_patch_y']-(cfg['fy'].shape[0]-1)*2+1
	cfg['wx'] = np.ones([1, cfg['len_wx']])
	cfg['wy'] = np.ones([cfg['len_wy'], 1])
	cfg['w'] = np.ones([cfg['len_wy'], cfg['len_wx']])

	######## OPTICAL PARAMETERS ########################
	# cfg['Sigma'] = 0.00029

	######## ASSERTION BEFORE RUNNING ##################
	# if the size of the image is odd number
	# the size of the valid patch should also be odd
	assert(np.mod(cfg['szx_sensor']-cfg['valid_patch_x'],2)==0)
	assert(np.mod(cfg['szy_sensor']-cfg['valid_patch_y'],2)==0)

if netName == "lensFlowNet":
	######## READ FROM FILES ########################
	cfg_file = "./opt_results/"+netName+"/1x1t-text34-setup0-filtered0-20-3x3lap"+".pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']
	cfg['szx_sensor'] = I.shape[2]
	cfg['szy_sensor'] = I.shape[1]

	# ######## DIFFERENTIAL FILTERS #####################
	# cfg['fx'] = np.array([[0.5,0,-0.5]])
	# cfg['fx'] = np.array(
	# 	[[0.5169,0.4626,0.4308,0.4325,0.2357,0,
	# 	-0.2357,-0.4325,-0.4308,-0.4626,-0.5169]]
	# )
	# cfg['fy'] = np.transpose(cfg['fx'])

	# cfg['fxx'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fx'],mode='full'
	# 	)
	# cfg['fyy'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fy'],mode='full'
	# 	)
	# cfg['fxy'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fy'],mode='full'
	# 	)
	# cfg['fyx'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fx'],mode='full'
	# 	)
	# cfg['fave'] = np.array([1,0])

	######## CONVOLUTIONAL WINDOW ######################
	cfg['valid_patch_x'] = 2
	cfg['valid_patch_y'] = cfg['valid_patch_x']

	cfg['separable'] = True
	cfg['len_wx'] = \
		cfg['szx_sensor']-cfg['valid_patch_x']-cfg['lap'].shape[1]+2
	cfg['len_wy'] = \
		cfg['szy_sensor']-cfg['valid_patch_y']-cfg['lap'].shape[0]+2
	cfg['wx'] = np.ones([1, cfg['len_wx']])
	cfg['wy'] = np.ones([cfg['len_wy'], 1])
	cfg['w'] = np.ones([cfg['len_wy'], cfg['len_wx']])

	# 

	# cfg['wx'] = np.ones([1,1])
	# cfg['wy'] = np.ones([1,1])
	# cfg['w'] = np.ones([1,1])


	# ######## OPTICAL PARAMETERS ########################
	# cfg['noise_var'] = 1e-5

	######## ASSERTION BEFORE RUNNING ##################
	# if the size of the image is odd number
	# the size of the valid patch should also be odd
	# assert(np.mod(cfg['szx_sensor']-cfg['valid_patch_x'],2)==0)
	# assert(np.mod(cfg['szy_sensor']-cfg['valid_patch_y'],2)==0)


if netName == "lensFlowNet1Df":
	######## READ FROM FILES ########################
	cfg_file = "./opt_results/"+netName+"/6"+".pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']

	# ######## DIFFERENTIAL FILTERS #####################
	# cfg['fx'] = np.array([[0.5,0,-0.5]])
	# cfg['fx'] = np.array(
	# 	[[0.5169,0.4626,0.4308,0.4325,0.2357,0,
	# 	-0.2357,-0.4325,-0.4308,-0.4626,-0.5169]]
	# )
	# cfg['fy'] = np.transpose(cfg['fx'])

	# cfg['fxx'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fx'],mode='full'
	# 	)
	# cfg['fyy'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fy'],mode='full'
	# 	)
	# cfg['fxy'] = signal.convolve2d(
	# 		cfg['fx'],cfg['fy'],mode='full'
	# 	)
	# cfg['fyx'] = signal.convolve2d(
	# 		cfg['fy'],cfg['fx'],mode='full'
	# 	)
	# cfg['fave'] = np.array([1,0])

	######## CONVOLUTIONAL WINDOW ######################
	cfg['valid_patch_x'] = 2
	cfg['valid_patch_y'] = cfg['valid_patch_x']

	cfg['separable'] = True
	cfg['len_wx'] = \
		cfg['szx_sensor']-cfg['valid_patch_x']-(cfg['fx'].shape[1]-1)*2+1
	cfg['len_wy'] = \
		cfg['szy_sensor']-cfg['valid_patch_y']-(cfg['fy'].shape[0]-1)*2+1

	cfg['wx'] = np.ones([1, cfg['len_wx']])
	cfg['wy'] = np.ones([cfg['len_wy'], 1])
	cfg['w'] = np.ones([cfg['len_wy'], cfg['len_wx']])


	# ######## OPTICAL PARAMETERS ########################
	# cfg['noise_var'] = 1e-5

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
	ff.input_images(I_input, Z_map_gt)
	
	# ff.regular_output()
	# cv2.waitKey(1)

	# Query some results for drawing
	query_list = ['Z_valid_flat', 'Z_gt_valid_flat','u_1_valid_flat','u_2_valid_flat'\
			,'u_3_valid_flat','u_4_valid_flat','conf_valid_flat','confw_valid_flat']
	res = ff.query_results(query_list)

	conf = res['conf_valid_flat']
	confw = res['confw_valid_flat']
	thre = 0.0

	# show how u_1 and u_2 are distributed 
	plt.figure()
	plt.subplot(221)
	plt.plot(res['u_1_valid_flat'][conf>thre],res['u_2_valid_flat'][conf>thre],'ro',ms=0.01)
	plt.xlabel('u_1')
	plt.ylabel('u_2')

	plt.subplot(222)
	plt.plot(res['u_3_valid_flat'][conf>thre],res['u_4_valid_flat'][conf>thre],'ro',ms=0.01)
	plt.xlabel('u_3')
	plt.ylabel('u_4')

	plt.subplot(223)
	plt.hist(res['u_1_valid_flat'][conf>thre],bins=100)
	plt.ylabel('u_1 number')
	plt.xlabel('u_1 frequency distribution')

	plt.subplot(224)
	plt.hist(res['u_2_valid_flat'][conf>thre],bins=100)
	plt.ylabel('u_2 number')
	plt.xlabel('u_2 frequency distribution')
	plt.show()

	# Update the derivative
	res['Z_valid_flat'] = res['Z_valid_flat'][confw>thre]
	res['Z_gt_valid_flat'] = res['Z_gt_valid_flat'][confw>thre]
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
	 yerr=[draw_list_std[:,0],draw_list_std[:,1]],fmt='ro')
plt.axis([min_depth, max_depth, min_depth, max_depth])
plt.ylabel('Estimated depth (m)')
plt.xlabel('True depth (m)')

plt.show()