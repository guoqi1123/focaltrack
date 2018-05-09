import tensorflow as tf
import numpy as np
import cv2 

import pdb


NAN_COLOR = np.array([0,0,255])
FONT_COLOR = (0,255,0)
""" prepares an image for debugging display 
		-ensures 3 color
		-eliminates NaNs, infities
		-rescales to 0 to 1, returns min/max values, will return nan,nan for invalid images (all non-finite)		
"""
def prep_for_draw(I, log = False, title = None, message = None, rng = [np.NaN, np.NaN]):
	#pdb.set_trace()
	valid = np.isfinite(I)
	invalid = ~valid
	
	if (not np.all(invalid)) and (not np.all(~np.isnan(rng))):
		# pdb.set_trace()
		rng = [np.min(I[valid]), np.max(I[valid])]

	#pdb.set_trace()
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
	
	#pdb.set_trace()
	#D = np.uint8(np.clip(255.0*D, 0, 255))
	if invalid.any():
		D[invalid] = NAN_COLOR
	
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
		
		T = np.zeros(tiled_shape, dtype=np.float32)
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


"""helper functions from the tensor flow docs"""
def make_kernel(a):
	a = np.asarray(a)
	a = a.reshape(list(a.shape) + [1,1])
	return tf.constant(a, dtype=tf.float32)

def simple_conv(x,k, padding = 'SAME'):
	x = tf.expand_dims(tf.expand_dims(x,0),-1)
	y = tf.nn.depthwise_conv2d(x,k,[1,1,1,1],padding=padding)
	return y[0,:,:,0]
	
def laplace(x):
	laplace_k = make_kernel([
		[0.5, 1.0, 0.5],
		[1.0,-6., 1.0],
		[0.5, 1.0, 0.5]
	])
	return simple_conv(x, laplace_k)
	
"""helper functions for Focal Flow"""
def dIdx(image, k = None):
	if k is None:
		k = [[0.5, 0.0, -0.5]]
	
	kernel = make_kernel(k)
	return simple_conv(image, kernel)

def dIdy(image, k = None):
	if k is None:
		k = [[0.5], [0.0], [-0.5]]
	kernel = make_kernel(k)
	return simple_conv(image, kernel)

def generate_window_kernels(sx = 35, sy = 35):
	W = tf.ones( (sy*2+1,sx*2+1) )
	#W_batched = tf.expand_dims(tf.expand_dims(W,0),-1) #TODO verify that this doesn't work
	W_batched = tf.expand_dims(tf.expand_dims(W,-1),-1)
	return W, W_batched

def separable_window(I, sx = 5, sy = 5, separable = True):
	# pdb.set_trace()
	wx = np.ones([sx, 1])
	wy = np.ones([1, sy])
	wx = wx.reshape(list(wx.shape) + [1,1])
	wy = wy.reshape(list(wy.shape) + [1,1])
	I = tf.expand_dims(tf.expand_dims(I,0),-1)  #fix shape for function
	if separable:
		return tf.nn.conv2d( 
			tf.nn.conv2d(I, wx, strides = [1,1,1,1], padding='SAME'),
			                wy, strides = [1,1,1,1], padding='SAME'
		)
	else:
		return tf.nn.conv2d(I, wx*wy, strides = [1,1,1,1], padding='SAME')

#useful regex: re.sub('\((\d+),(\d+),\:\)', "['\g<1>\g<2>']",y)	
			
""" takes in a dictionary of the different elements of the matrix with keys as strings:
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

def test_4x4():
	for i in range(2):
		X = 1000.0*np.random.rand(4,4)
		#enforce symmetry
		for i,j in [(1,2), (1,3), (2,3), (1,4), (2,4), (3,4)]:
			X[j-1][i-1] = X[i-1][j-1]		
		
		X_inv = np.linalg.inv(X)
		
		if np.linalg.matrix_rank(X) < 4:
			print("!!! generated matrix with rank < 4, skipping!!!")
			continue
		
		X_dict = {}
		for i in range(4):
			for j in range(4):
				X_dict['%d%d'%(i+1,j+1)] = X[i][j]
		X_dict_inv, X_det = inv4x4(X_dict)
		X_inv2 = np.zeros((4,4))
		
		for i in range(4):
			for j in range(4):
				X_inv2[j][i] = X_dict_inv['%d%d'%(i+1,j+1)]
				
		sse1 = np.sum((np.eye(4) - np.dot(X,X_inv))**2.0)/16.0
		sse2 = np.sum((np.eye(4) - np.dot(X,X_inv2))**2.0)/16.0
		error = (X_inv - X_inv2)**2.0
		
		print("X")
		print(X)
		print("X_inv")
		print(X_inv)
		
		numeric_det = np.linalg.det(X)
		print("numeric det %2.2e, cofactors det %2.2e, error %2.2e"%(numeric_det, X_det, numeric_det/X_det))
		
		print("X_inv2")
		print(X_inv2)
		print("|X_inv - X_inv2|")
		print(np.abs(X_inv - X_inv2))
		print("\n")
		print("sse1 = ", sse1, "sse2 = ", sse2, "ratio = ", sse1/sse2)
		print("error")
		print(error)
		print("X*X_inv")
		print(np.dot(X,X_inv))
		print("X*X_inv2")
		print(np.dot(X,X_inv2))
		
		print("\n\n")
		
if __name__ == "__main__":
	test_4x4()






























