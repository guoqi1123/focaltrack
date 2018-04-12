cfg_file = "./opt_results/"+netName+"/2"+".pickle"
	with open(cfg_file,'rb') as f:
		cfg_data = pickle.load(f)
	cfg = cfg_data['cfg']