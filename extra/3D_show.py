# demonstrate the 3D models
import numpy as np
import pickle
import json
import os, glob
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

def valid_region_I(cfg, I):
	resolution = [[cfg[i]['szy_sensor'], cfg[i]['szx_sensor']] for i in range(len(cfg))]
	rows_cut = int(\
		((cfg[-1]['gauss'].shape[0]-1)/2+\
		(cfg[-1]['ext_f'].shape[1]-1)/2)*\
		resolution[0][0]/resolution[-1][0]
	)
	cols_cut = int(\
		((cfg[-1]['gauss'].shape[1]-1)/2+\
		(cfg[-1]['ext_f'].shape[2]-1)/2)*\
		resolution[0][1]/resolution[-1][1]
	)

	rows = cfg[0]['szx_sensor']
	cols = cfg[0]['szy_sensor']

	I = \
		I[
			cols_cut:cols-cols_cut,
			rows_cut:rows-rows_cut
		]
	return I

# depth_file = "./models/"+\
# 		"121.pickle" #74 start of shield #88 start of take 2 with shield
depth_file = "./models/quad/"+\
		"125.pickle"

with open(depth_file,'rb') as f:
	depth_data = pickle.load(f)

x = np.arange(depth_data['Zf'].shape[1])
y = np.arange(depth_data['Zf'].shape[0])
[X,Y] = np.meshgrid(x,y)

# read data
I_0 = depth_data['I_0']
I_1 = depth_data['I_1']
Z = depth_data['Zf']
conf = depth_data['conff']

# set the depth range
dmin = -2.0
dmax = -0.8
Z[np.where(Z <= dmin)] = np.NaN
Z[np.where(Z >= dmax)] = np.NaN



cfg_file = "./opt_results/pyConfLensFlowNetFast_ext/"+\
		"1x1t-text34-py4-setup5-one-sequential-regularize-nothreshold.pickle"
with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']
I_0 = valid_region_I(cfg[0:-1], I_0)
I_1 = valid_region_I(cfg[0:-1], I_1)

# draw the captured image
fig = plt.figure()
ax = fig.add_subplot(1,3,1, title="First image")
plt.imshow(I_0, cmap = 'gray',vmin=0, vmax=255)
plt.axis('off')
ax = fig.add_subplot(1,3,2, title="Second image")
plt.imshow(I_1, cmap = 'gray',vmin=0, vmax=255)
plt.axis('off')
# plt.imshow(conf, cmap= 'gray')

# confidence threshold
conf_thre = [0.999]
for i in range(len(conf_thre)):
	Z_cut = Z
	Z_cut[np.where(conf<= conf_thre[i])] = np.NaN
	# Z_cut[np.where(Z<=dmin)] = np.NaN
	# Z_cut[np.where(Z>=dmax)] = np.NaN

	ax = fig.add_subplot(1,3,i+3, title="Depth map, conf > "+str(conf_thre[i]), projection='3d')
	Z_cut = Z_cut[::-1]
	# plt.imshow(Z_cut, cmap='jet', origin='low', vmin=dmin, vmax=dmax)
	Z_cut[np.where(Z_cut>dmax)] = np.NaN
	Z_cut[np.where(Z_cut<dmin)] = np.NaN
	xx,yy = np.meshgrid(np.arange(Z_cut.shape[1]),np.arange(Z_cut.shape[0]))
	pdb.set_trace()
	ax.plot_surface(xx,yy,Z_cut)
	# plt.colorbar()
	plt.axis('off')
plt.show()

