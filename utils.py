import tensorflow as tf
import numpy as np
import cv2
import pdb
from scipy import signal
from skimage.transform import pyramid_laplacian
from skimage.transform import pyramid_gaussian
import copy
import scipy.misc 
import matplotlib.pyplot as plt



NAN_COLOR = np.array([0,0,0])
FONT_COLOR = (0,1.0,0)
########################################################
# All the utility functions
########################################################
"""helper functions from the tensor flow docs"""
def make_kernel(a):
	return tf.expand_dims(tf.expand_dims(a, -1), -1)

def simple_conv(x,k, padding = 'SAME'):
	x = tf.expand_dims(tf.expand_dims(x,0),-1)
	y = tf.nn.depthwise_conv2d(x,k,[1,1,1,1],padding=padding)
	return y[0,:,:,0]

def simple_conv_batch(x,k, padding = 'SAME'):
	x = tf.expand_dims(x,-1)
	y = tf.nn.depthwise_conv2d(x,k,[1,1,1,1],padding=padding)
	return y[:,:,:,0]
	
def dIdx(image, k = None):
	if k is None:
		k = tf.Variable([[0.5, 0.0, -0.5]],dtype=tf.float32)
	
	kernel = make_kernel(k)
	return simple_conv(image, kernel)

def dIdy(image, k = None):
	if k is None:
		k = tf.Variable([[0.5], [0.0], [-0.5]],dtype=tf.float32)
	kernel = make_kernel(k)
	return simple_conv(image, kernel)
	
def dIdx_batch(image, k = None):
	if k is None:
		k = [[0.5, 0.0, -0.5]]
	kernel = make_kernel(k)
	return simple_conv_batch(image, kernel)

def dIdy_batch(image, k = None):
	if k is None:
		k = [[0.5], [0.0], [-0.5]]
	kernel = make_kernel(k)
	return simple_conv_batch(image, kernel)

def dIdt(image, k = None, padding = 'SAME'):
	if k is None:
		k = tf.Variable([[[-0.5,0,0.5]]],dtype=tf.float32)
	k = tf.expand_dims(k,-1)
	x = tf.expand_dims(image,0)
	y = tf.nn.conv2d(x, k, strides =[1,1,1,1], padding = padding)
	return y[0,:,:,0]

def separable_window(I, w_x, w_y, padding = 'SAME'):
	# pdb.set_trace()
	wx = tf.expand_dims(tf.expand_dims(w_x,-1),-1)
	wy = tf.expand_dims(tf.expand_dims(w_y,-1),-1)
	I = tf.expand_dims(tf.expand_dims(I,0),-1)
	return tf.nn.conv2d( 
		tf.nn.conv2d(I, wx, strides = [1,1,1,1], padding=padding),
		                wy, strides = [1,1,1,1], padding=padding
	)

def unseparable_window(I, w, padding = 'SAME'):
	w = tf.expand_dims(tf.expand_dims(w,-1),-1)
	I = tf.expand_dims(tf.expand_dims(I,0),-1)
	return tf.nn.conv2d(I, w, strides = [1,1,1,1], padding=padding)

#useful regex: re.sub('\((\d+),(\d+),\:\)', "['\g<1>\g<2>']",y)	
			
""" takes in a dictionary of the different elements of the matrix 
with keys as strings:
	[ [ 11 12 13 14 ] 
	  [ 21 22 23 24 ]
	  [ 31 32 33 34 ]
	  [ 41 42 43 44 ] ]	  

"""
def det4x4(M):
	detM = \
		M['11']*M['22']*M['33']*M['44'] + \
		M['11']*M['23']*M['34']*M['42'] + \
		M['11']*M['24']*M['32']*M['43'] + \
		\
		M['12']*M['21']*M['34']*M['43'] + \
		M['12']*M['23']*M['31']*M['44'] + \
		M['12']*M['24']*M['33']*M['41'] + \
		\
		M['13']*M['21']*M['32']*M['44'] + \
		M['13']*M['22']*M['34']*M['41'] + \
		M['13']*M['24']*M['31']*M['42'] + \
		\
		M['14']*M['21']*M['33']*M['42'] + \
		M['14']*M['22']*M['31']*M['43'] + \
		M['14']*M['23']*M['32']*M['41'] - \
		\
		M['11']*M['22']*M['34']*M['43'] - \
		M['11']*M['23']*M['32']*M['44'] - \
		M['11']*M['24']*M['33']*M['42'] - \
		\
		M['12']*M['21']*M['33']*M['44'] - \
		M['12']*M['23']*M['34']*M['41'] - \
		M['12']*M['24']*M['31']*M['43'] - \
		\
		M['13']*M['21']*M['34']*M['42'] - \
		M['13']*M['22']*M['31']*M['44'] - \
		M['13']*M['24']*M['32']*M['41'] - \
		\
		M['14']*M['21']*M['32']*M['43'] - \
		M['14']*M['22']*M['33']*M['41'] - \
		M['14']*M['23']*M['31']*M['42']
	return detM

"""
	inverts 4x4 symmetric matrix
	
	takes in a dictionary of the different elements of the matrix with keys as strings:
	[ [ 11 12 13 14 ] 
	  [ 21 22 23 24 ]
	  [ 31 32 33 34 ]
	  [ 41 42 43 44 ] ]	  
	NOTE: ASSUMES SYMMETRY:
	[ [ 11 12 13 14 ] 
	  [ 12 22 23 24 ]
	  [ 13 23 33 34 ]
	  [ 14 24 34 44 ] ]
"""

def inv4x4sym(M):

	detM = det4x4(M)
	
	B = {}
	B['11'] = \
		  M['22'] * M['33'] * M['44'] \
		+ M['23'] * M['34'] * M['42'] \
		+ M['24'] * M['32'] * M['43'] \
		- M['22'] * M['34'] * M['43'] \
		- M['23'] * M['32'] * M['44'] \
		- M['24'] * M['33'] * M['42'];

	B['12'] = \
		  M['12'] * M['34'] * M['43'] \
		+ M['13'] * M['32'] * M['44'] \
		+ M['14'] * M['33'] * M['42'] \
		- M['12'] * M['33'] * M['44'] \
		- M['13'] * M['34'] * M['42'] \
		- M['14'] * M['32'] * M['43'];

	B['13'] = \
		  M['12'] * M['23'] * M['44'] \
		+ M['13'] * M['24'] * M['42'] \
		+ M['14'] * M['22'] * M['43'] \
		- M['12'] * M['24'] * M['43'] \
		- M['13'] * M['22'] * M['44'] \
		- M['14'] * M['23'] * M['42'];

	B['14'] = \
		  M['12'] * M['24'] * M['33'] \
		+ M['13'] * M['22'] * M['34'] \
		+ M['14'] * M['23'] * M['32'] \
		- M['12'] * M['23'] * M['34'] \
		- M['13'] * M['24'] * M['32'] \
		- M['14'] * M['22'] * M['33'];

	B['22'] = \
		  M['11'] * M['33'] * M['44'] \
		+ M['13'] * M['34'] * M['41'] \
		+ M['14'] * M['31'] * M['43'] \
		- M['11'] * M['34'] * M['43'] \
		- M['13'] * M['31'] * M['44'] \
		- M['14'] * M['33'] * M['41'];

	B['23'] = \
		  M['11'] * M['24'] * M['43'] \
		+ M['13'] * M['21'] * M['44'] \
		+ M['14'] * M['23'] * M['41'] \
		- M['11'] * M['23'] * M['44'] \
		- M['13'] * M['24'] * M['41'] \
		- M['14'] * M['21'] * M['43'];

	B['24'] = \
		  M['11'] * M['23'] * M['34'] \
		+ M['13'] * M['24'] * M['31'] \
		+ M['14'] * M['21'] * M['33'] \
		- M['11'] * M['24'] * M['33'] \
		- M['13'] * M['21'] * M['34'] \
		- M['14'] * M['23'] * M['31'];

	B['33'] = \
		  M['11'] * M['22'] * M['44'] \
		+ M['12'] * M['24'] * M['41'] \
		+ M['14'] * M['21'] * M['42'] \
		- M['11'] * M['24'] * M['42'] \
		- M['12'] * M['21'] * M['44'] \
		- M['14'] * M['22'] * M['41'];

	B['34'] = \
		  M['11'] * M['24'] * M['32'] \
		+ M['12'] * M['21'] * M['34'] \
		+ M['14'] * M['22'] * M['31'] \
		- M['11'] * M['22'] * M['34'] \
		- M['12'] * M['24'] * M['31'] \
		- M['14'] * M['21'] * M['32'];

	B['44'] = \
		  M['11'] * M['22'] * M['33'] \
		+ M['12'] * M['23'] * M['31'] \
		+ M['13'] * M['21'] * M['32'] \
		- M['11'] * M['23'] * M['32'] \
		- M['12'] * M['21'] * M['33'] \
		- M['13'] * M['22'] * M['31'];

	B['21'] = B['12'];
	B['31'] = B['13'];
	B['32'] = B['23'];
	B['41'] = B['14'];
	B['42'] = B['24'];
	B['43'] = B['34'];
	
	C = {}
	for k in B.keys():	
		C[k] = B[k]/(1.0*detM)
	return C, B, detM

""" Methods for display """
def prep_for_draw(I, log = False, title = None, message = None, rng = [np.NaN, np.NaN]):
	#pdb.set_trace()
	if len(I.shape) == 2 or I.shape[2] == 1:
		valid = np.isfinite(I)
		invalid = ~valid
		
		if (not np.all(invalid)) and (not np.all(~np.isnan(rng))):
			# pdb.set_trace()
			rng = [np.min(I[valid]), np.max(I[valid])]

		#convert to color image
		D = I.copy()
		
		D = np.float32(D)

		if len(I.shape) == 2 or I.shape[2] == 1:
			D = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)
		if D.shape[2] != 3:
			raise Exception("Unsupported shape for prepping for draw: " + str(D.shape))	
		
		D = (D-rng[0])/float(rng[1] - rng[0])
		
		if log:
			D = np.log10(D)
			D_rng = [np.min(D), np.max(D)]
			D = (D-D_rng[0])/float(D_rng[1] - D_rng[0])
		
		#D = np.uint8(np.clip(255.0*D, 0, 255))
		if invalid.any():
			D[invalid] = NAN_COLOR
	else:
		D = I
	
	# D[I > rng[1]] = NAN_COLOR
	# D[I < rng[0]] = NAN_COLOR
	
	t_s = 0.7
	t_h = int(20*t_s)
	title_str = "[%1.1e,%2.1e]"%(rng[0], rng[1])
	if title is not None:
		title_str = title  + "::" + title_str
	
	cv2.putText(D, title_str, (0,t_h), cv2.FONT_HERSHEY_DUPLEX, t_s, FONT_COLOR)
	
	if message is not None:
		#message = "fps: %2.2f"%self.get_fps()
		cv2.putText(D, message, (0, D.shape[0]-t_h), cv2.FONT_HERSHEY_DUPLEX, t_s, FONT_COLOR)
	return D

def tile_image(I, rng = None, labels = None, log = False, title = None, message = None):
	if isinstance(I, dict):
		keys = sorted(I.keys())
		values = [I[k] for k in keys]
		if rng == None:
			rng = [[np.NaN, np.NaN] for k in keys]
		else:
			rng_val = [rng[k] for k in keys]
		return tile_image( \
					I = values, \
					rng = rng_val, \
					labels = keys, \
					log = log, \
					title = title, \
					message = message\
				)
	if isinstance(I, list):	
		#try to assemble square tilings, 
		R = int(np.sqrt(len(I))) 
		C = int(np.ceil(len(I)/R))
	
		shape = I[0].shape
		y = shape[0]
		x = shape[1]
		tiled_shape = np.zeros(3)
		tiled_shape[0:2] = np.array([R,C])*np.array(shape)
		tiled_shape[2] = 3
		
		T = np.zeros(tiled_shape.astype(np.int), dtype=np.float32)
		processed = 0
		r = 0
		c = 0
		while processed < len(I):
			label  = "%d/%d"%(processed+1, len(I))
			if labels is not None:
				label = labels[processed]
			if title is not None:
				label = "%s:%s"%(title, label)
			T[r*y:(r+1)*y, c*x:(c+1)*x] = prep_for_draw(\
												I[processed], \
												log = log, \
												title = label, \
												message = label, \
												rng = rng[processed]
											) #TODO replace this 
			processed += 1
			c += 1
			if c >= C:
				c = 0
				r += 1
				if r >= R:
					r = 0
		return T
	raise Exception("Tiling failure")	

""" Adding noise to the image"""
def gauss_noise(I, mu = 0, var = 1e-5):
	return I + np.random.normal(mu, np.sqrt(var), I.shape)

""" Create Gaussian and laplacian pyramid"""
def create_pyramid(I, layer_num=5, downscale=2):
	# create a image pyramid
	I_py_temp = tuple(pyramid_gaussian(\
		I[0,:,:,:], \
		max_layer=layer_num-1, \
		downscale=downscale
	))
	I_lap_py_temp = tuple(pyramid_laplacian(\
		I[0,:,:,:], \
		max_layer=layer_num-1, \
		downscale=downscale
	))
	I_py = []
	I_lap_py = []
	for i in range(layer_num):
		I_py.append(
			np.empty(
				(
					(I.shape[0],)+
					I_py_temp[i].shape
				)
			)
		)
		I_lap_py.append(
			np.empty(
				(
					(I.shape[0],)+
					I_lap_py_temp[i].shape
				)
			)
		)

	for i in range(I.shape[0]):
		I_py_temp = tuple(pyramid_gaussian(\
			I[i,:,:,:], \
			max_layer=layer_num-1, \
			downscale=downscale
		))
		I_lap_py_temp = tuple(pyramid_laplacian(\
			I[i,:,:,:], \
			max_layer=layer_num-1, \
			downscale=downscale
		))
		for j in range(layer_num):
			I_py[j][i,:,:,:] = I_py_temp[j]
			I_lap_py[j][i,:,:,:] = I_lap_py_temp[j]
	
	return I_py, I_lap_py

def create_pyramid_tuple(I, layer_num=5, downscale=2):
	# return the pyramid as a list of tuples
	I_py = []
	I_lap_py = []
	for i in range(I.shape[0]):
		I_py.append(
			tuple(pyramid_gaussian(\
				I[i,:,:,:], \
				max_layer=layer_num-1, \
				downscale=downscale
			))
		)
		I_lap_py.append(
			tuple(pyramid_laplacian(\
				I[i,:,:,:], \
				max_layer=layer_num-1, \
				downscale=downscale
			))
		)
	return I_py, I_lap_py

""" band pass filters"""
def bp_filter_batch(I, low=0, high=10):
	I_back = I
	for i in range(I.shape[0]):
		for j in range(I.shape[3]):
			I_back[i,:,:,j] = bp_filter(I[i,:,:,j],low, high)
	return I_back

def bp_filter(I, low=0, high=10):
	# only keep the part with [low, high]
	I_f = np.fft.fft2(I)
	I_f = np.fft.fftshift(I_f)
	rows = I_f.shape[0]
	cols = I_f.shape[1]
	c_row = int(rows/2)
	c_col = int(cols/2)

	I_f[c_row-low+1:c_row+low,c_col-low+1:c_col+low] = 0
	I_f1 = copy.deepcopy(I_f)
	I_f1[c_row-high:c_row+high+1,c_col-high:c_col+high+1] = 0
	I_f = I_f - I_f1

	I_f = np.fft.ifftshift(I_f)
	I_back = np.fft.ifft2(I_f)
	I_back = np.abs(I_back)

	# I_disp = I_back - I_back.min()
	# I_disp = I_disp / I_disp.max()
	# cv2.imshow("1", I_disp)
	# cv2.waitKey(1)
	return I_back

def resize(I, rows, cols):
	I_back = np.zeros((I.shape[0],rows,cols,I.shape[3]))
	for i in range(I.shape[0]):
		for j in range(I.shape[3]):
			I_back[i,:,:,j] = scipy.misc.imresize(\
				I[i,:,:,j],(rows, cols)
			)

			I_disp = I_back[i,:,:,j] - I_back.min()
			I_disp = I_disp / I_disp.max()
			cv2.imshow("1", I_disp)
			cv2.waitKey(1)

	return I_back