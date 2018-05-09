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
conf_thre = [0.99]

data_name = ['one_new','one_accommodation']
legend_name = {
	'one_new'	:	'Without accommodation',
	'one_accommodation':	'Accommodation'
}

base = [0,0.9,0.99,0.999,0.9999]
xs = np.array([i/10**(j+1)+base[j] for i in range(0,10) for j in range(len(base))])
ys = np.array([0.001 for i in range(0,10) for j in range(len(base))])
conf_range = -np.log((1 - np.array([0,0.9999])))
data = {}
file_dir = "./test_results/pyConfLensFlowNetFast_ext/"
for name in data_name:
	with open(file_dir+name+'.pickle','rb') as f:
		data[name] = pickle.load(f)

fig = plt.figure()	
# draw the sparsification curve
ax1 = fig.add_subplot(3,2,1)
legends = []
conf_wr = {}
for name in data_name:
	# get the data
	Z_flat = data[name]['Z_flat'].flatten()
	Z_gt_flat = data[name]['Z_gt_flat'].flatten()
	conf_flat = data[name]['conf_flat'].flatten()

	# data is too much, sample them
	idx = ((np.random.rand(500000)*len(Z_flat)).astype(np.int),)
	Z_flat = Z_flat[idx]
	Z_gt_flat = Z_gt_flat[idx]
	conf_flat = conf_flat[idx]
	
	err = np.abs(Z_flat - Z_gt_flat).astype(np.float64)

	# sort the conf_flat
	err_sorted = err[np.argsort(conf_flat)]
	conf_sorted = np.sort(conf_flat)
	sparse = np.arange(len(err))/len(err)
	num = len(err) - np.arange(len(err))

	for i in range(len(err)-1,0,-1):
		err_sorted[i-1] += err_sorted[i]
	err_sorted /= num

	# draw a fig that shows the average error with a certain sparsication
	bin_nums = 1000
	step = np.linspace(0, len(err_sorted)-1, bin_nums, True).astype(np.int)
	
	err_show = err_sorted[step]
	sparse_show = sparse[step]
	conf_show = conf_sorted[step]
	step_wr = np.linspace(0, bin_nums-1, 6, True).astype(np.int)
	# conf_wr[name] = conf_show[step_wr]
	conf_wr[name] = np.array([0,0.9,0.99,0.999,0.9999])

	# compute the AUC
	area = np.mean(err_sorted)

	# draw the figure	
	conf_show_log = -np.log((1 - conf_show))
	line, = ax1.plot(conf_show_log, err_show, '-', label=legend_name[name])
	ax1.get_xaxis().set_visible(False)
	if name == data_name[-1]:
		ax1.plot(-np.log((1 - xs)),ys,'k|')
		# fig.canvas.draw()
		# labels = [float(item.get_text()) for item in ax1.get_xticklabels()]
		# labels = 1-np.exp(-np.array(labels))
		# labels = [('%.5f' % label).rstrip('0').rstrip('.') for label in labels]
		# labels = []
		# ax1.set_xticklabels(labels)
		plt.xlim((conf_range.min(),conf_range.max()))
		ax1.get_xaxis().set_visible(False)

	legends.append(line)

# ax1.set_xlabel('Confidence level')
# ax1.set_ylabel('Mean error (m)')
plt.ylim((0.0,0.14))
# plt.xlim((0.9,1.0))

# show the legend separately
ax1 = fig.add_subplot(3,2,6)
plt.legend(handles=legends)

# # draw the mean error at each depth for a certain confidence level
# for i in range(len(conf_per)):
# 	ax2 = fig.add_subplot(3,2,i+2, title='Top '+str(int(conf_per[i]*100))+'%')
# 	for name in data_name:
# 		# get the data
# 		off = 1.38
# 		# cut Z_gt_flat to 2 digits
# 		Z_gt_flat = data[name]['Z_gt_flat'][0,:] + off
# 		Z_gt_flat = (Z_gt_flat * 100).astype(np.int32)/100
# 		Z_gt_unique = np.unique(Z_gt_flat)

# 		# select top data from Z
# 		Z = data[name]['Z_flat']+off
# 		conf = data[name]['conf_flat']
# 		idx_sort = np.argsort(-conf,0)
# 		num_cut = int(conf_per[i]*idx_sort.shape[0])
# 		idx_sort = idx_sort[0:num_cut,:]
# 		xx, yy = np.meshgrid(np.arange(idx_sort.shape[1]),np.arange(idx_sort.shape[0]))
# 		idx_sort = (idx_sort,xx)
# 		Z_per = Z[idx_sort]
# 		err_per = np.abs(Z_per - np.tile(np.expand_dims(Z_gt_flat,0),[num_cut,1]))
# 		err_per_mean = np.mean(err_per,0)

# 		# compute average error for a ground truth depth
# 		err_means = []
# 		for Z_gt in Z_gt_unique:
# 			Z_gt_idx = np.where(Z_gt_flat==Z_gt)
# 			err_mean = np.mean(err_per_mean[Z_gt_idx])		
# 			err_means.append(err_mean)
# 		err_means = np.array(err_means)	
# 		pdb.set_trace()	

# 		ax2.plot(Z_gt_unique, err_means)
# 		ax2.set_xlabel('True depth (m)')
# 		ax2.set_ylabel('Mean error (m)')
# 		plt.ylim((0,0.4))
# 		plt.xlim((0.5,0.9))

# draw the mean error at each depth for a certain confidence level
for i in range(len(conf_thre)):
	ax2 = fig.add_subplot(3,2,i+3, title='Conf >'+str(conf_thre[i]))
	for name in data_name:
		# get the data
		off = 1.38

		# get the data
		Z_flat = data[name]['Z_flat'].flatten()
		Z_gt_flat = data[name]['Z_gt_flat'].flatten()
		conf_flat = data[name]['conf_flat'].flatten()

		# data is too much, sample them
		idx = ((np.random.rand(500000)*len(Z_flat)).astype(np.int),)
		Z_flat = Z_flat[idx]
		Z_gt_flat = Z_gt_flat[idx]
		
		err_flat = np.abs(Z_flat-Z_gt_flat)
		conf_flat = conf_flat[idx]
		idx_sort = np.argsort(-conf_flat,0)
		num_cut = len(conf_flat[np.where(conf_flat>conf_thre[i])])
		idx_sort = idx_sort[0:num_cut]
		conf_sort = conf_flat[(idx_sort,)]
		err_sort = err_flat[(idx_sort,)]
		Z_gt_sort = Z_gt_flat[(idx_sort,)]
		Z_gt_sort = (Z_gt_sort * 100).astype(np.int32)/100
		Z_gt_unique = np.unique(Z_gt_sort)

		# compute average error for a ground truth depth
		err_means = []
		for Z_gt in Z_gt_unique:
			Z_gt_idx = np.where(Z_gt_sort==Z_gt)
			err_mean = np.mean(err_sort[Z_gt_idx])		
			err_means.append(err_mean)
		err_means = np.array(err_means)	

		ax2.plot(Z_gt_unique+off, err_means)
		ax2.plot(Z_gt_unique+off,0.1*(Z_gt_unique+off),'k--')
		# ax2.set_xlabel('True depth (m)')
		# ax2.set_ylabel('Mean error (m)')
		plt.ylim((0,0.14))
		plt.xlim((0.26,1.12))

plt.show()


# compute the working range
ax3 = fig.add_subplot(3,2,4)

wr = {}
for name in data_name:
	wr[name] = []

	# get the data
	off = 1.38

	# 
	Z_flat = data[name]['Z_flat'].flatten()
	Z_gt_flat = data[name]['Z_gt_flat'].flatten()
	conf_flat = data[name]['conf_flat'].flatten()
	err_flat = np.abs(Z_flat-Z_gt_flat)

	for i in range(len(conf_wr[name])):
		# get the data
		idx_sort = np.where(conf_flat>conf_wr[name][i])
		conf_flat = conf_flat[idx_sort]
		err_flat = err_flat[idx_sort]
		Z_gt_flat = Z_gt_flat[idx_sort]
		Z_gt_flat = (Z_gt_flat * 100).astype(np.int32)/100
		Z_gt_unique = np.unique(Z_gt_flat)

		# compute average error for a ground truth depth
		err_means = []
		for Z_gt in Z_gt_unique:
			Z_gt_idx = np.where(Z_gt_flat==Z_gt)
			err_mean = np.mean(err_flat[Z_gt_idx])		
			err_means.append(err_mean)
		err_means = np.array(err_means)	

		err_thre = 0.1*(Z_gt_unique+off)
		wr_idx = np.where(err_thre>err_means)[0]
		if len(wr_idx)==0:
			wr[name].append(np.nan)
		else:
			wr[name].append(Z_gt_unique[wr_idx].max()-Z_gt_unique[wr_idx].min())

	# draw the figure	
	conf_show_log = -np.log((1 - conf_wr[name]))
	ax3.plot(conf_show_log, wr[name], '-')
	ax3.get_xaxis().set_visible(False)
	if name == data_name[-1]:
		ax3.plot(-np.log((1 - xs)),ys,'k|')
		# fig.canvas.draw()
		# labels = [float(item.get_text()) for item in ax3.get_xticklabels()]
		# labels = 1-np.exp(-np.array(labels))
		# labels = [('%.5f' % label).rstrip('0').rstrip('.') for label in labels]
		# labels = []
		# ax3.set_xticklabels(labels)
		plt.xlim((conf_range.min(),conf_range.max()))
		ax3.get_xaxis().set_visible(False)
# 
ax4 = fig.add_subplot(3,2,2)
for name in data_name:
	# get the data
	conf_flat = data[name]['conf_flat'].flatten()

	# data is too much, sample them
	idx = ((np.random.rand(500000)*len(conf_flat)).astype(np.int),)

	conf_sorted = np.sort(conf_flat[idx])
	sparse = np.arange(len(conf_sorted))/len(conf_sorted)

	# draw a fig that shows the average error with a certain sparsication
	bin_nums = 1000
	step = np.linspace(0, len(conf_sorted)-1, bin_nums, True).astype(np.int)
	
	conf_show = conf_sorted[step]
	sparse_show = sparse[step]

	# draw the figure	
	conf_show_log = -np.log((1 - conf_show))
	line, = ax4.plot(conf_show_log, sparse_show, '-', label=legend_name[name])
	ax4.get_xaxis().set_visible(False)
	if name == data_name[-1]:
		ax4.plot(-np.log((1 - xs)),ys,'k|')
		# fig.canvas.draw()
		# labels = [float(item.get_text()) for item in ax4.get_xticklabels()]
		# labels = 1-np.exp(-np.array(labels))
		# labels = [('%.5f' % label).rstrip('0').rstrip('.') for label in labels]
		# labels = []
		# ax4.set_xticklabels(labels)
		plt.xlim((conf_range.min(),conf_range.max()))
		ax4.get_xaxis().set_visible(False)

# ax4.set_xlabel('Confidence level')
# ax4.set_ylabel('Sparsity')

# show the legend separately
ax1 = fig.add_subplot(3,2,6)
plt.legend(handles=legends)


plt.show()