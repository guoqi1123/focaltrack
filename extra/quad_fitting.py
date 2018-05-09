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
		"quad1.pickle"

with open(depth_file,'rb') as f:
	depth_data = pickle.load(f)

# set the depth range
dmin = -2.0
dmax = -0
Z = depth_data['Z']
offset = depth_data['offset']
Z_filt = []
blur = np.ones((10,10))
blur = blur/np.sum(blur)
import scipy.signal
pxs = []
pys = []
for i in range(Z.shape[0]):
	Z_filt.append(scipy.signal.convolve2d(Z[i,:,:], blur, mode='valid'))

	# fig = plt.figure()
	# ax = fig.add_subplot(1,2,1,projection='3d')
	# xx, yy = np.meshgrid(np.arange(Z_filt[i].shape[1]),np.arange(Z_filt[i].shape[0]))
	# ax.plot_surface(xx, yy,-Z_filt[i],vmin=dmin, vmax=dmax)
	# ax = fig.add_subplot(1,2,2)
	# ax.imshow(-Z[i,:,:], vmin=dmin, vmax=dmax)
	# plt.show()

fig = plt.figure()
for i in range(Z.shape[0]):
	x = np.arange(Z_filt[i].shape[1])
	x = x - (Z_filt[i].shape[1]-1)/2

	ax = fig.add_subplot(1,2,1)
	Z_x = np.mean(Z_filt[i],0)
	Z_x = (Z_x + np.flipud(Z_x))/2
	order= 4
	px = np.polyfit(x,Z_x,order)
	pxs.append(px/px[-1])
	print(px/px[-1])
	val = np.stack([px[k]*x**(order-k) for k in range(order+1)],-1)
	val = np.sum(val, -1)
	ax.plot(x,Z_x)
	ax.plot(x, val)

	y = np.arange(Z_filt[i].shape[0])
	y = y - (Z_filt[i].shape[0]-1)/2
	
	ax = fig.add_subplot(1,2,2)
	Z_y = np.mean(Z_filt[i],1)
	Z_y = (Z_y + np.flipud(Z_y))/2
	order = 4
	py = np.polyfit(y,Z_y,order)
	print(py/py[-1])
	pys.append(py/py[-1])
	val = np.stack([py[k]*y**(order-k) for k in range(order+1)],-1)
	val = np.sum(val, -1)
	ax.plot(y,Z_y)
	ax.plot(y, val)
	# plt.show()

pxs = np.stack(pxs, 0)
pys = np.stack(pys, 0)
orders = 4
x = offset/10000
ppx = []
ppy = []
for k in range(pxs.shape[1]):
	ppx.append(np.polyfit(x,pxs[:,k],orders))
for k in range(pys.shape[1]):
	ppy.append(np.polyfit(x,pys[:,k],orders))

fig = plt.figure()	
valx = []
for i in range(pxs.shape[1]):
	ax = fig.add_subplot(2,5,i+1)
	plt.plot(x, pxs[:,i],'.')
	val = np.stack([ppx[i][k]*x**(orders-k) for k in range(orders+1)],-1)
	val = np.sum(val, -1)
	plt.plot(x, val)
	valx.append(val)

valy = []
for i in range(pys.shape[1]):
	ax = fig.add_subplot(2,5,i+6)
	plt.plot(x, pys[:,i],'.')
	val = np.stack([ppy[i][k]*x**(orders-k) for k in range(orders+1)],-1)
	val = np.sum(val, -1)
	plt.plot(x, val)	
	valy.append(val)

plt.show()

# validation 
for i in range(Z.shape[0]):
	x = offset[i]/10000
	# quadratic correction
	px = []
	py = []
	for j in range(len(ppx)):
		o = np.stack([x**k for k in range(len(ppx[j]))],0)
		px.append(np.sum(np.flipud(ppx[j]) * o))

	for j in range(len(ppy)):
		o = np.stack([x**k for k in range(len(ppy[j]))],0)
		py.append(np.sum(np.flipud(ppy[j]) * o))

	z = Z[i,:,:]
	xx, yy = np.meshgrid(np.arange(z.shape[1]),np.arange(z.shape[0]))
	xx = (xx - (z.shape[1]-1)/2)
	yy = (yy - (z.shape[0]-1)/2)

	xr = [xx**(len(px)-k-1) * px[k] for k in range(len(px))]
	xr = np.sum(np.stack(xr,0),0)
	yr = [yy**(len(py)-k-1) * py[k] for k in range(len(py))]
	yr = np.sum(np.stack(yr,0),0)

	zr = xr * yr

	# also makes a soft threshold for confidence
	pxd = np.array([px[k]*(len(px)-1-k) for k in range(len(px)-1)])
	pyd = np.array([py[k]*(len(py)-1-k) for k in range(len(py)-1)])
	rxs = np.roots(pxd)
	rys = np.roots(pyd)

	# confx = 1/(1+np.exp(0.2*(np.abs(xx)-rxs.max())))
	# confy = 1/(1+np.exp(0.2*(np.abs(yy)-rxs.max())))
	# confr = confx * confy

	# confx = 1/(1+np.exp(200*(np.abs(xx)/rxs.max()-1)))
	# confy = 1/(1+np.exp(200*(np.abs(yy)/rys.max()-1)))
	# confr = confx * confy
	xxr = np.abs(xx)/rxs.max()
	yyr = np.abs(yy)/rxs.max()
	confr = 1/(1+np.exp(200*(np.sqrt(xxr**2+yyr**2)-1)))
	z[np.where(confr<0.999)] = np.nan

	fig = plt.figure()
	ax = fig.add_subplot(1,2,1,projection='3d')
	ax.plot_surface(xx, yy, -z/zr, vmin=dmin, vmax=dmax)
	ax = fig.add_subplot(1,2,2)
	plt.imshow(-z/zr, vmin = dmin, vmax=dmax)
	plt.show()



cfg_file = "./opt_results/pyConfLensFlowNetFast_iccv5/"+\
		"final.pickle"
with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']
for i in range(len(cfg)):
	cfg[i]['ppx'] = ppx
	cfg[i]['ppy'] = ppy
cfg_data['cfg'] = cfg


with open(cfg_file,'wb') as f:
	pickle.dump(cfg_data, f)
