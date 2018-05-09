import pickle
import json
import os, glob
import matplotlib.pyplot as plt
import numpy as np
import pdb

# data_name = ['initial', 'half', 'one', 'two', 'sparse']
# legend_name = {
# 	'initial':	'Initialization',
# 	'half'	: 	'0.5-norm',
# 	'one'	: 	'1-norm',
# 	'two'	: 	'2-norm',
# 	'sparse':	'AUSC',
# }

data_name = ['one_new','one_accommodation']
legend_name = {
	'one_new'	:	'1-norm',
	'one_accommodation':	'Accommodation'
}

data = {}
file_dir = "./test_results/pyConfLensFlowNetFast_ext/"
for name in data_name:
	with open(file_dir+name+'.pickle','rb') as f:
		data[name] = pickle.load(f)

# fig = plt.figure()	
# # draw the sparsification curve
# ax1 = fig.add_subplot(3,2,1, title="Sparsification Curve")
# legends = []
# for name in data_name:
# 	# get the data
# 	Z_flat = data[name]['Z_flat'].flatten()
# 	Z_gt_flat = data[name]['Z_gt_flat'].flatten()
# 	conf_flat = data[name]['conf_flat'].flatten()

# 	# data is too much, sample them
# 	idx = ((np.random.rand(500000)*len(Z_flat)).astype(np.int),)
# 	Z_flat = Z_flat[idx]
# 	Z_gt_flat = Z_gt_flat[idx]
# 	conf_flat = conf_flat[idx]
	
# 	err = np.abs(Z_flat - Z_gt_flat).astype(np.float64)

# 	# sort the conf_flat
# 	err_sorted = err[np.argsort(conf_flat)]
# 	sparse = np.arange(len(err))/len(err)
# 	num = len(err) - np.arange(len(err))

# 	for i in range(len(err)-1,0,-1):
# 		err_sorted[i-1] += err_sorted[i]
# 	err_sorted /= num

# 	# draw a fig that shows the average error with a certain sparsication
# 	bin_nums = 1000
# 	step = np.linspace(0, len(err_sorted)-1, bin_nums, True).astype(np.int)
	
# 	err_show = err_sorted[step]
# 	sparse_show = sparse[step]

# 	# compute the AUC
# 	area = np.mean(err_sorted)

# 	# draw the figure	
# 	line, = ax1.plot(sparse_show, err_show, '-', label=legend_name[name])
# 	legends.append(line)

# plt.legend(handles=legends)
# ax1.set_xlabel('Sparsification')
# ax1.set_ylabel('Average error')
# plt.ylim((0.0,0.14))


# # draw the AUC vs. depth
# i = 1
# for name in data_name:
# 	i += 1
# 	ax2 = fig.add_subplot(3,2,i, title=legend_name[name])
# 	# get the data
# 	Z_gt_flat = data[name]['Z_gt_flat'][0,:] + 1.38
# 	AUC = data[name]['AUC']
# 	ax2.plot(Z_gt_flat, AUC, '.')
# 	ax2.set_xlabel('True depth (m)')
# 	ax2.set_ylabel('AUSC of each prediction')
# 	plt.ylim((0,0.4))
# 	plt.xlim((0.25,1.2))

# draw average AUC vs. depth
i = 1
legends = []
fig = plt.figure()
ax3 = fig.add_subplot(1,1,i)
for name in data_name:
	Z_gt_flat = data[name]['Z_gt_flat'][0,:] + 1.38
	AUC = data[name]['AUC']
	Z_gt_flat_uni = np.unique(Z_gt_flat)
	AUC_mean = []
	for Z_gt_elem in Z_gt_flat_uni:
		AUC_mean.append(
			np.mean(
				AUC[np.where(Z_gt_flat==Z_gt_elem)]
			)
		)
	line, = ax3.plot(Z_gt_flat_uni, AUC_mean, '.', label=legend_name[name])
	legends.append(line)
	ax3.set_xlabel('True depth (m)')
	ax3.set_ylabel('Average AUSC (m)')
	plt.ylim((0,0.3))
	plt.xlim((0.26,1.12))

# plt.legend(handles=legends)
ax3.plot([0,10],[0.05,0.05],'k')
plt.show()