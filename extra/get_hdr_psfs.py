#!/usr/bin/python2.7

import cv2
import numpy as np
import flycapture2 as fc2
import scipy.misc
import serial 
ser = serial.Serial()

# physical settings
#exposures = [20, 10, 5, 2.5, 1.25, .625, .3125, .15625, .078125, .0390625, .01953125, .009765625] # ms
exposures = [.005, .01, .02, .04, .08, .16, .32, .64, 1.28, 2.56, 5.12, 10.24, 20.48] # ms
mufs = range(5000,65000,10000) # cam goes 0 to 65535
deltamuf =1600 # twice amplitude 
lightdist = "1p5ft"

# connect
ser.port = "/dev/ttyUSB0"
ser.baudrate = 9600
ser.open()


for muf in mufs:

	for pulse in [0,1]:

		path = "./hdrims_diffuse/"+lightdist+"_"+str(muf+pulse*deltamuf).replace(".","p")+"muf_"
		path2 = "./hdrims_diffuse/hdr/"+lightdist+"_"+str(muf+pulse*deltamuf).replace(".","p")+"muf"

		# set lens
		if ser.isOpen():
			string = ["AM0;","TD8197;","FR50;","OF" + str(muf+pulse*deltamuf) + ";"]
			#["AM8000;","TD8197;","FR50;","OF34904;"]
			for i in range(len(string)):
			    ser.write(string[i].encode())
			    response = ser.read(ser.inWaiting())
			    print(response)

		# connect to cam
		c = fc2.Context()
		c.connect(*c.get_camera_from_index(0))
		c.start_capture()

		# get ims
		for exp in exposures:
			# set exposure
			#print("Current shutter settings: %s\n\n" %c.get_property(fc2.SHUTTER))
			shutter={'abs_control': True, 'one_push': False, 'value_a': 256, 'type': 12, 'auto_manual_mode': False, 'present': True, 'abs_value': exp, 'value_b': 0, 'on_off': True}
			c.set_property(**shutter)
			# take im
			im = fc2.Image()
			c.retrieve_buffer(im)
			#img = np.clip((1-np.array(im))*255, 0, 255).astype('uint8')
			img = (1-np.array(im))*255
			# save im
			scipy.misc.imsave(path+str(exp).replace(".","p")+"ms.png",(img).astype(np.uint8))
			
		# make and save HDR im
		img_fn = [path+str(s).replace(".","p")+"ms.png" for s in exposures]
		img_list = [cv2.imread(fn) for fn in img_fn]

		#calibrate = cv2.createCalibrateDebevec()
		#response = calibrate.process(img_list,np.array(exposures,dtype=np.float32).copy())

		merge_debvec = cv2.createMergeDebevec()
		hdr_debvec = merge_debvec.process(img_list,np.array(exposures,dtype=np.float32).copy())#,response)
		#cv2.imwrite(path2+".png",hdr_debvec*255)
		tonemap1 = cv2.createTonemap(gamma=1.0)
		res_debvec = tonemap1.process(hdr_debvec.copy())
		cv2.imwrite(path2+".png",res_debvec*255)#/res_debvec.max())
		##res_debvec_8bit = res_debvec*255/res_debvec.max()
		#res_debvec_8bit = np.clip(res_debvec*255/res_debvec.max(), 0, 255).astype('uint8')
		#cv2.imwrite(path2+".png",res_debvec_8bit)

