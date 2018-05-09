# This code simulates the focal flow camera
# Author: Qi Guo, Harvard University
# Email: qguo@seas.harvard.edu
# All Rights Reserved

import numpy as np
from numpy import linalg as LA
from scipy import interpolate
from scipy import stats
from scipy import signal
from scipy.stats import multivariate_normal
# import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
import json
import os
import pdb
import pickle

class FFCamera(object):
	"""a class for camera"""
	def __init__(self, cfg = {}):
		"default setting"
		self.cfg = {
			#Camera parameters
			'psf_func':		"PSF_gauss", #psf function model
			'Sigma':		.001, 	#Standard deviation of the aperture
									#MUST BE ISOTROPIC
			'pix_size':		5.86e-6 * 2, #all in meters
			'f':			1e-1, #focal distance
			'mu_s':			1.3e-1,
			'szx_sensor':	350,
			'szy_sensor':	350,
		}

		for k in cfg.keys():
			self.cfg[k] = cfg[k]

		# Location of principle points
		self.cfg['x_prinpts'] = self.cfg['szx_sensor']/2
		self.cfg['y_prinpts'] = self.cfg['szy_sensor']/2
		# Compute mu_f
		self.cfg['mu_f'] = 1./(1./self.cfg['f']-1./self.cfg['mu_s'])

	def set_texture(self, img_cfg):
		"This function set the texture to simulate"
		# default image configuration
		self.img_cfg = {
			# parameter of the texture, assume the texture are infinite
			# (round padding)
			'text_paths': 		[],
			'pix_in_m':		0.0002,  # how big is a pixel in the texture
			# parameter of the movement
			#
			'v_dir':			np.array([0.,0.,-2.]), # directions of movement
										  # between frames, it will be normalized 
			'v_len':			.001,	#speed m/frame
			'z_range':  		np.array([0.260,0.600]),
			'hf_seq':			1, # hf_seq * 2 + 1 is the number of frames 
								   # generated in a row
			#
			# 
			'exp_num':			100, # number of experiments
		}
		for k in img_cfg.keys():
			self.img_cfg[k] = img_cfg[k]

		self.img_cfg['v_dir'] = \
			self.img_cfg['v_dir']/np.sqrt(sum(self.img_cfg['v_dir']**2.))

	def im2double(im):
	    mat = cvGetMat(im);
	    if CV_MAT_DEPTH(mat.type)==CV_64F:
	       return mat
	    im64f = array(size(im), 'double')
	    cvConvertScale(im, im64f, 1.0, 0.0)
	    return im64f

	def simulation(self):
		"This function conducts simulation"
		for texture in self.img_cfg['text_paths']:
			self.texture = cv2.imread(texture)
			if len(self.texture.shape) == 3:
				# if RGB, turns it into gray
				self.texture = cv2.cvtColor(self.texture, cv2.COLOR_RGB2GRAY)

			# convert to float
			self.texture = cv2.normalize(
				self.texture.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX
			)
			self.img_cfg['res'] = self.texture.shape

			# XX and YY are x and y coordinates of each pixel of the
			# texture plane. We assume that the center of the texture
			# (on optical axis) is the upper right. Corner of the 
			# image and we do round padding, so the texture is 
			# actually infinite, 
			XX,YY = np.meshgrid(
						np.arange(self.img_cfg['res'][1]),\
						np.arange(self.img_cfg['res'][0])
			)
			pts_t = np.concatenate(
				(
					np.reshape(XX,(-1,1)),
					np.reshape(YY,(-1,1))
				),
				axis = 1
			)
			text_1d = np.reshape(self.texture,(-1,1))

			# simulation starts
			self.I = np.empty(
				(
					self.img_cfg['exp_num'],
					self.cfg['szy_sensor'],
					self.cfg['szx_sensor'],
					self.img_cfg['hf_seq']*2+1,
				)
			)
			self.Loc = np.empty((self.img_cfg['exp_num'],3,self.img_cfg['hf_seq']*2+1))
			for i in range(self.img_cfg['exp_num']):
				loc = np.random.rand(3)
				# Randomly generate the location and depth
				x_center = loc[0] * self.img_cfg['res'][1] * self.img_cfg['pix_in_m']
				y_center = loc[1] * self.img_cfg['res'][0] * self.img_cfg['pix_in_m']
				z_center = \
					loc[2]*(self.img_cfg['z_range'][1]-self.img_cfg['z_range'][0])\
					+self.img_cfg['z_range'][0]
				# Fix the location and depth for debugging
				x_center = self.img_cfg['res'][1] * self.img_cfg['pix_in_m'] / 2
				y_center = self.img_cfg['res'][0] * self.img_cfg['pix_in_m'] / 2
				z_center = 0.250 + i * 0.004
				
				for j in np.arange(
					-self.img_cfg['hf_seq'],
					self.img_cfg['hf_seq']+1
				):
					# compute the center location of texture
					x = x_center + \
						j*self.img_cfg['v_dir'][0]*self.img_cfg['v_len']
					y = y_center + \
						j*self.img_cfg['v_dir'][1]*self.img_cfg['v_len']
					z = z_center + \
						j*self.img_cfg['v_dir'][2]*self.img_cfg['v_len']
					# magnification
					mag = self.cfg['mu_s']*self.img_cfg['pix_in_m']\
							/z /self.cfg['pix_size']
					print("Magnification: ",mag)
					
					# use different sampling technique for different 
					# magnification
					if mag >= 1:
						# generate psf
						psf_x, psf_y = eval(\
							'self.'+self.cfg['psf_func']+'(z)'
						)
						self.I[i,:,:,j+self.img_cfg['hf_seq']]\
						= self.sample_then_blur(
							pts_t, text_1d, mag, psf_x, psf_y, x, y
						)
					else:
						# generate psf
						psf_x, psf_y = eval(\
							'self.'+self.cfg['psf_func']+'(z,'\
							+'\'texture\',mag)'
						)
						self.I[i,:,:,j+self.img_cfg['hf_seq']]\
						= self.blur_then_sample(
							pts_t, self.texture, mag, psf_x, psf_y, x, y
						)
					self.Loc[i,:,j+self.img_cfg['hf_seq']]= [x,y,z]
				# The beneath code is for debugging
				for k in range(self.img_cfg['hf_seq']*2+1):
					cache1 = self.I[i,:,:,k]
					cv2.imshow("Generated Image", cache1)
					cv2.waitKey(10)
				print("Depth of the central image: ",z)
				
			self.write_to_files(texture)

	def sample_then_blur(self,pts_t,text_1d,mag,psf_x,psf_y,x,y):
		"""
		This function first sample from textures to obtain sharp images
		and then blur it using PSF, using it when mag >= 1
		"""
		# propagate the coordinate of each sensor pixel onto the 
		# texture plane
		X_s = (np.arange(self.cfg['szx_sensor']\
			+psf_x.shape[1]-1+psf_y.shape[1]-1)\
			-self.cfg['x_prinpts']-(psf_x.shape[1]-1)/2\
			-(psf_y.shape[1]-1)/2)/mag\
			-x/self.img_cfg['pix_in_m']
		Y_s = (np.arange(self.cfg['szy_sensor']\
			+psf_x.shape[0]-1+psf_y.shape[0]-1)\
			-self.cfg['y_prinpts']-(psf_y.shape[0]-1)/2\
			-(psf_x.shape[0]-1)/2)/mag\
			-y/self.img_cfg['pix_in_m']

		# As we are using round padding, we need to mod the X_s
		# and Y_s to make them within the range of XX and YY
		X_s = np.remainder(X_s, self.img_cfg['res'][1]-1)
		Y_s = np.remainder(Y_s, self.img_cfg['res'][0]-1)

		X_s, Y_s = np.meshgrid(
			X_s, Y_s
		)
		pts_s = np.concatenate(
			(
				np.reshape(X_s,(-1,1)),
				np.reshape(Y_s,(-1,1))
			),
			axis = 1
		)
		# the sharp image captured by camera can be approximated
		# as the interpolation of the sensor coordinates onto
		# the texture coordinate map
		P = interpolate.griddata(pts_t, text_1d, pts_s, method = 'linear')
		P = np.reshape(P,X_s.shape)
		pdb.set_trace()
		# We then convolve the sharp image with the blur kernel
		temp = signal.convolve2d(P,psf_x,mode='valid')
		return signal.convolve2d(temp,psf_y,mode='valid')

	def blur_then_sample(self,pts_t,text_2d,mag,psf_x,psf_y,x,y):
		"""
		This function first blur the image using PSF, then sample it to
		obtain the final image, using it when mag < 1
		"""
		# We first convolve the sharp image with the magnified blur kernel
		temp = signal.convolve2d(
			text_2d,psf_x,mode='same',boundary ='wrap'
		)
		temp = signal.convolve2d(
			temp,psf_y,mode='same',boundary ='wrap'
		)
		text_1d = np.reshape(temp,(-1,1))
		# propagate the coordinate of each sensor pixel onto the 
		# texture plane
		X_s = (np.arange(self.cfg['szx_sensor'])\
			-self.cfg['x_prinpts'])/mag\
			-x/self.img_cfg['pix_in_m']
		Y_s = (np.arange(self.cfg['szy_sensor'])\
			-self.cfg['y_prinpts'])/mag\
			-y/self.img_cfg['pix_in_m']
		# As we are using round padding, we need to mod the X_s
		# and Y_s to make them within the range of XX and YY
		X_s = np.remainder(X_s, self.img_cfg['res'][1]-1)
		Y_s = np.remainder(Y_s, self.img_cfg['res'][0]-1)

		X_s, Y_s = np.meshgrid(
			X_s, Y_s
		)
		pts_s = np.concatenate(
			(
				np.reshape(X_s,(-1,1)),
				np.reshape(Y_s,(-1,1))
			),
			axis = 1
		)
		# the sharp image captured by camera can be approximated
		# as the interpolation of the sensor coordinates onto
		# the texture coordinate map
		I = interpolate.griddata(pts_t, text_1d, pts_s, method = 'linear')
		return np.reshape(I, X_s.shape)


	def write_to_files(self,texturePath):
		"This function write to files for each texture"
		base = os.path.basename(texturePath)
		base = os.path.splitext(base)
		textureName = base[0] # Find the name of the texture
		fileName = os.path.join(\
			"simulation_data/focalFlowNet",textureName+".pickle"
		)
		# cut the text_paths to have only one texture path in it
		img_cfg_temp = self.img_cfg
		img_cfg_temp['text_paths'] = [texturePath]


		with open(fileName,'wb') as f:
			data = {
				'cfg'		:self.cfg,
				'img_cfg'	:img_cfg_temp,
				'I'			:self.I,
				'Loc'		:self.Loc,
			}
			# dump the data into the file
			pickle.dump(data, f)
				
	def PSF_gauss(self, z, plane = 'sensor', mag = 1):
		"Generate a separable gaussian PSF, only Gaussian without rotation"
		# if we would like to compute the PSF on texture plane,
		# the magnification from sensor to texture need to be considered
		if plane == 'sensor':
			mag = 1
		mu_f = self.cfg['mu_f']
		mu_s = self.cfg['mu_s']
		Sigma = self.cfg['Sigma']
		pixSize = self.cfg['pixSize']

		mag_Sigma = np.abs(1./z - 1./mu_f) * mu_s / mag
		Cov = (Sigma * mag_Sigma / pixSize) ** 2
		lbda = [Cov, Cov]
		# Aligned Covariance
		Cov_aligned = np.diag(lbda)
		# Standard deviation to determine the size of patch
		lbda_std = np.sqrt(lbda)
		# 3 is deterimined using standard normal distribution chart
		sz = (np.ceil(lbda_std * 3) * 2 + 1).astype(np.int)
		if sz[0] < 10:
			sz = np.array([10,10])
		xx = np.arange(sz[1]) - (sz[1]-1)/2
		yy = np.arange(sz[0]) - (sz[0]-1)/2

		if LA.det(Cov_aligned) == 0:
			# If the PSF vanishes to a single point
			return 1,1
		else:
			rv_x = multivariate_normal([0],lbda[0])
			rv_y = multivariate_normal([0],lbda[1])
		psf_x = rv_x.pdf(xx)
		psf_y = rv_y.pdf(yy)
		psf_x = psf_x / np.sum(psf_x)
		psf_y = psf_y / np.sum(psf_y)
		return np.array([psf_x]), \
				np.transpose(np.array([psf_y]))

psf_func = "PSF_gauss"
camParam = {
	'psf_func':psf_func, #psf function model
	'Sigma':0.0010001, #Std of the isotropic filter
	'pixSize':5.86e-6 * 2, #all in meters
	'f':1e-1,
	'mu_s':1.3e-1,
	'szx_sensor':350,
	'szy_sensor':350,
}
camera1 = FFCamera(camParam)
imgParam = {
	'text_paths': \
		[
			"Qi_curetgrey/0001.png",
			# "Qi_curetgrey/0398.png",
			# "Qi_curetgrey/0724.png",
			# "Qi_curetgrey/1956.png",
			# "Qi_curetgrey/2469.png",
			# "Qi_curetgrey/3048.png",
			# "Qi_curetgrey/4987.png",
		],
}
camera1.set_texture(imgParam)
cv2.namedWindow("Generated Image", cv2.WINDOW_NORMAL)
camera1.simulation()