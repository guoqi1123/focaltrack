import argparse

import cv2
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.signal
import pickle
import json
import os, glob
import math
import copy

from utils import *
import time
from scipy import interpolate

cfg_file = "./params/base.pickle"
with open(cfg_file,'rb') as f:
	cfg_data = pickle.load(f)
cfg = cfg_data['cfg']

for i in range(len(cfg)):
	cfg[i]['b0'] = 1e-4
	cfg[i]['b1'] = -1.9

cfg_file = "./params/final.pickle"
with open(cfg_file,'rb') as f:
	cfgf_data = pickle.load(f)
cfgf = cfgf_data['cfg']

for i in range(len(cfgf)):
	cfgf[i]['a1'] = 0
	cfgf[i]['ra1_1'] = 0
	cfgf[i]['ra1_2'] = 0

data = {}
data['cfg'] = cfg
data['cfgf'] = cfgf
with open("./params/default.pickle",'wb') as f:
	pickle.dump(data, f)
pdb.set_trace()