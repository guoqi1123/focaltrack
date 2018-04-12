import pickle
import json
import pdb
import numpy as np

fileName = "./experiment_data/pyConfLensFlowNetFast/9-0002g.pickle"
with open(fileName,'rb') as f:
	data = pickle.load(f)

# data['offset'] = np.ones((data['I'].shape[0],))*34904
pdb.set_trace()

with open(fileName,'wb') as f:
	# dump the data into the file
	pickle.dump(data, f)