# transform the focal flow data into standard format for training
import pickle
import json
import os, glob
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.io as spio
import pdb

matfile = './experiment_data/focalFlowNet_multi_mu_s/test_data/test.mat'
f = h5py.File(matfile, 'r')
I = np.transpose(np.array(f.get('I')),(3,2,1,0))
cfg = {
	'psf_func'		:	'apodizing',
	'Sigma'			:	np.array(f.get('Sigma'))[0,0],
	'pix_size'		:	np.array(f.get('pix_size'))[0,0],
	'f'				:	np.array(f.get('f'))[0,0],
	'szx_sensor'	:	np.array(f.get('szx_sensor'))[0,0],
	'szy_sensor'	:	np.array(f.get('szy_sensor'))[0,0],
	'num_ave'		:	np.array(f.get('num_ave'))[0,0],
	'Z_0'			:	np.array(f.get('Z_0'))[0,0],
}
Loc = np.transpose(np.array(f.get('Loc')),(2,1,0))
mu_s = np.reshape(np.array(f.get('mu_s')),-1)

# part the data to save
directory = './experiment_data/focalFlowNet_multi_mu_s/'
os.chdir(directory)
lpickle = len(glob.glob('*.pickle'))
num = 80
st = 0
end = np.min([I.shape[0],num+st])
for i in range(20):
	fileName = os.path.join(str(i+lpickle)+".pickle")
	with open(fileName, 'wb') as f:
		data = {
			'cfg'	:	cfg,
			'I'		:	I[st:end,:,:,:],
			'Loc'	:	Loc[st:end,:,:],
			'mu_s'	:	mu_s[st:end],
		}
		pickle.dump(data, f)
	st = end
	end = np.min([I.shape[0],num+st])
