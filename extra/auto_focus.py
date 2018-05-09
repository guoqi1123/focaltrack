import serial 

import argparse

import cv2
import flycapture2 as fc2
import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import json
import os, glob

import matplotlib.pyplot as plt
from training_pyConfLensFlowNetFast import training_pyConfLensFlowNetFast

from utils import *
import time

#notes
#  -did I mix up x and y again?  always worth checking

#debugging
import pdb

#multithreading
import threading
myThread = threading.Condition()
I_cache = 0
I_in_input = 0
I_lap_batch_input = 0
input_dict = 0
ending_key = 'c'


class Camera():
	"""A class for the camera"""
	def __init__(self, P, dP):
		#default configuraiton
		self.cfg = {
			#processing configuration knobs
			'wrapper': 'ptg', # Select the wrapper
			'downscale': 2, #downscaling of the image
			'camera' : 0, #the integer mapping to the camera you 
			#want to use.  typically 0, but depends on what's available
			'input_video' : None, #file name for a video file to 
			#use for debugging.  Will just loop through.  Can also 
			#use something like "path-to-images/%2d.tif" as the 
			#input here, that will show through all the images 
			#that match that string.
		}

		self.t = []
		self.draw = None
		self.cache = {}
		self.vars = {}
		self.frames = 0
		self.resolution = None

		#update index
		self.P = P
		self.dP = dP
		self.var = 0

		# the port
		self.ser = serial.Serial()
		self.ser.port = "/dev/ttyUSB0" # may be called something different
		self.ser.baudrate = 9600 # may be different
		self.ser.open()
		
		#initialize camera
		self.cam = None
		self.initialize_camera_ptg()

		#initialize cache
		for k in ['gray']:
			self.cache[k] = np.zeros(self.resolution)
		
		#initialize computation
		self.Z_t = []

		self.sample_images = None

		self.data_type = tf.float32

		lf = 96
		up = 176
		self.roi = [lf,self.resolution[0]-2*lf,up,self.resolution[1]-2*up]
		self.I_cache = np.zeros([self.roi[1],self.roi[3]], dtype = np.uint8)

	def run(self):
		if self.ser.isOpen():
			string = "OF"+str(self.P)+";"
			self.ser.write(string.encode())
			response = self.ser.read(self.ser.inWaiting())
		self.grab_frame_and_process_ptg()
		self.var = np.var(self.I_cache)

		while True:
			if self.ser.isOpen():
				string = "OF"+str(self.P+self.dP)+";"
				self.ser.write(string.encode())
				response = self.ser.read(self.ser.inWaiting())

			self.grab_frame_and_process_ptg()

			self.regular_output()
			c = cv2.waitKey(100) & 0xFF

			var_new = np.var(self.I_cache)
			if var_new > self.var:
				self.var = var_new
				self.P += self.dP
				self.dP *= 2
				self.dP = int(self.dP)
				self.P = int(self.P)
				print("P: ", self.P, ", Var: ", self.var)
			else:
				self.dP = -self.dP
				self.dP = int(self.dP)
				self.P = int(self.P)

				if self.ser.isOpen():
					string = "OF"+str(self.P+self.dP)+";"
					self.ser.write(string.encode())
					response = self.ser.read(self.ser.inWaiting())

				self.grab_frame_and_process_ptg()

				self.regular_output()
				c = cv2.waitKey(100) & 0xFF

				if var_new > self.var:
					self.var = var_new
					self.P += self.dP
					self.dP *= 2
					self.dP = int(self.dP)
					self.P = int(self.P)
					print("P: ", self.P, ", Var: ", self.var)
				else:
					self.dP = self.dP/2
					self.dP = int(self.dP)
					self.P = int(self.P)			

			if np.abs(self.dP) < 1:
				print("In focus P: ", self.P)
				return

		
	def initialize_camera_ptg(self):
		#access the point grey camera using flycap
		print(fc2.get_library_version())
		self.ptg = fc2.Context()
		print(self.ptg.get_num_of_cameras())
		self.ptg.connect(*self.ptg.get_camera_from_index(0))
		print(self.ptg.get_camera_info())

		m, f = self.ptg.get_video_mode_and_frame_rate()
		print(m, f)

		# print(c.get_video_mode_and_frame_rate_info(m, f))
		print(self.ptg.get_property_info(fc2.FRAME_RATE))
		p = self.ptg.get_property(fc2.FRAME_RATE)
		print(p)
		self.ptg.set_property(**p)
		self.ptg.start_capture()

		self.im = fc2.Image()
		# try to get the first image to 
		print([np.array(self.ptg.retrieve_buffer(self.im)).sum() for i in range(80)])
		img = np.array(self.im)
		print(img.shape, img.base)
		
		# downscaling the images
		img = scipy.misc.imresize(img, 1/self.cfg['downscale'])
		self.resolution = (
			img.shape[0],img.shape[1]
		)
		self.cfg['camera_fps'] = p['abs_value']

	"""imports a frame into """
	def process(self, new_frame):
		self.cache['raw'] = new_frame
		if len(new_frame.shape) > 2:
		# or new_frame.shape[2] != 1:
			self.cache['gray'] = cv2.cvtColor(self.cache['raw'], cv2.COLOR_BGR2GRAY)
		else:
			self.cache['gray'] = self.cache['raw'].copy()
		
		self.I_cache[:,:] = self.cache['gray'][\
			self.roi[0]:self.roi[0]+self.roi[1],
			self.roi[2]:self.roi[2]+self.roi[3],
		]
		
		return
		
	"""helper: grabs the frame and runs the processing stage""" 
	def grab_frame_and_process_ptg(self):
		# Using the point grey camera
		self.ptg.retrieve_buffer(self.im)
		img = np.array(self.im)
		img = scipy.misc.imresize(img, 1/self.cfg['downscale']).astype(np.float32)
		for i in range(20):
			self.ptg.retrieve_buffer(self.im)
			img += scipy.misc.imresize(np.array(self.im), 1/self.cfg['downscale'])
		img /= 21
		return self.process(img)
			
	"""computes the average FPS over the last __FPS_WINDOW frames"""
	def get_fps(self):
		__FPS_WINDOW = 4
		if len(self.t) < __FPS_WINDOW:
			return -1
		else:
			return __FPS_WINDOW/(1.0*sum(self.t[-1:-1 - __FPS_WINDOW:-1]))

	"""computes some performance statistics at the end"""
	def final_statistics(self):
		if len(self.t) < 2:
			print("too short, invalid statistics...")
			return

		t = np.array(self.t)
		fps = 1.0/t;
		print("""
			#####################################
			### CAMERA CAPTURE FPS STATISTICS ###
			#####################################
			  min: %(min)f
			  med: %(median)f
			  avg: %(avg)f
			  max: %(max)f
			#####################################
			"""%{'min':fps.min(), 'avg':fps.mean(), 'median': np.median(fps), 'max': fps.max()})

	def regular_output(self):
		self.cache['draw'] = tile_image(\
									[self.I_cache.astype(np.float32)], \
									rng = [[0,255]], \
									log = False, \
									title = "Camera Thread", \
								)
		cv2.imshow("Camera Thread", self.cache['draw'])
			
	"""destructor: free up resources when done"""
	def __del__(self):
		#TODO video writer stuff here
		cv2.destroyAllWindows()


a = Camera(37000,1000)
a.run()

