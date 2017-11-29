import numpy as np 
import glob

def get_local_files():
	'''
	Load data files from local directory
	'''

	data_dir = "<path>/CANDLE/Benchmarks.git/Pilot2/common/generate_datasets"
	data_files = glob.glob('%s/*.npz'%data_dir)

	import pilot2_datasets as p2
	fields = p2.gen_data_set_dict()

	return (data_files, fields)

def get_data_arrays(f):

	data = np.load(f)

	X = data['features']
	nbrs = data['neighbors']
	resnums = data['resnums']

	return (X, nbrs, resnums)

def append_nbrs(x, nbrs, num_nbrs):

	new_x_shape = np.array(x.shape)
	new_x_shape[1] *= num_nbrs+1 
	x_wNbrs = np.zeros(new_x_shape)

	for i in range(len(x)):
	    nb_indices = nbrs[i, :num_nbrs+1].astype(int)
	    nb_indices = nb_indices[nb_indices!=-1]

	    temp_mols = x[nb_indices]
	    newshape = (1,np.prod(temp_mols.shape))
	    temp_mols = np.reshape(temp_mols,newshape)

	    x_wNbrs[i,:temp_mols.shape[1]] = temp_mols

	return x_wNbrs
