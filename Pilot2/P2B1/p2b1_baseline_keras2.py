import theano
import numpy as np
import scipy as sp
import pickle
import sys,os
import glob
import optparse
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..','..', 'common'))
sys.path.append(lib_path)
from data_utils import get_file
HOME=os.environ['HOME']
def parse_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

if __name__=="__main__":
### Hyperparameters and model save path
	parser=optparse.OptionParser()
	parser.add_option("--train", action="store_true",dest="train_bool",default=True,help="Invoke training")
	parser.add_option("--evaluate", action="store_true",dest="eval_bool",default=False,help="Use model for inference")
	parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='.')
	parser.add_option("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
	parser.add_option("--config-file",help="Config File",dest="config_file",type=str,default=os.path.join(file_path, 'p2b1_small_model.txt'))
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	parser.add_option("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
	parser.add_option("--data-set",help="[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]",dest="set_sel",type=str,default="3k_Disordered")
	(opts,args)=parser.parse_args()


	if not os.path.isdir(opts.home_dir):
		print ('Keras home directory not set')
		sys.exit(0)
	sys.path.append(opts.home_dir)
	
	import p2b1 as hf
	reload(hf)
	
	lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
	sys.path.append(lib_path)

	import keras_model_utils as KEU
	reload(KEU)
	import pilot2_datasets as p2
	reload(p2)
	maps=hf.autoencoder_preprocess()
	
	from keras.optimizers import SGD,RMSprop,Adam
	from keras.datasets import mnist
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras import callbacks
	from keras.layers.advanced_activations import ELU
	from keras.preprocessing.image import ImageDataGenerator

	GP=hf.ReadConfig(opts.config_file)
	batch_size = GP['batch_size']
	
##### Read Data ########
	data_set=p2.data_sets[opts.set_sel][0]
	data_hash=p2.data_sets[opts.set_sel][1]
	print ('Reading Data Files... %s->%s' % (opts.set_sel, data_set))
        data_file = get_file(data_set, origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot2/'+data_set+'.tar.gz', untar=True, md5_hash=data_hash)
        data_dir = os.path.join(os.path.dirname(data_file), data_set)
	data_files=glob.glob('%s/*.npy'%data_dir) 
	
	## Define datagenerator
	datagen=hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])  

	## get data dimension ##
	num_samples = 0
	for f in data_files:
		X=np.load(f)
		num_samples += X.shape[0]

	X=np.load(data_files[0])
	print 'Data Format: [Num Sample (%s), Num Molecules (%s), Num Atoms (%s), Position + Molecule Tag (One-hot encoded) (%s)]' % (
          num_samples, X.shape[1], X.shape[2], X.shape[3])

	X_train=hf.get_data(X,case=opts.case)
	input_dim=X_train.shape[1]
	
### Define Model, Solver and Compile ##########
	print ('Define the model and compile')
	opt = Adam(lr=GP['learning_rate'])
	
	print ('using mlp network')
	model_type='mlp'
	hidden_layers=GP['num_hidden']
	model=hf.dense_auto(weights_path=opts.weight_path,input_shape=(input_dim,),nonlinearity='elu',\
		hidden_layers=hidden_layers,l2_reg=GP['weight_decay'])
	memo='%s_%s'%(opts.base_memo,model_type)

	print 'Autoencoder Regression problem'
	model.compile(optimizer=opt, loss='mean_squared_error')

#### Print Model Stats ###########
	KEU.Model_Info(model)
	
#### Train the Model
	if opts.train_bool:
		if not str2bool(GP['cool']):
			effec_epochs=GP['epochs']
			ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=opts.case)
			loss=ct.train_ac()
		else:
			effec_epochs=GP['epochs']//3
			ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=opts.case)
			loss=[]
			for i in range(3):
				lr=GP['learning_rate']/10**i
				ct.model.optimizer.lr.set_value(lr)
				if i>0:
					ct.print_data=False
					print 'Cooling Learning Rate by factor of 10...'
				loss.extend(ct.train_ac())
				
		if opts.save_path!=None:
			loss_file='%s/%s.pkl'%(opts.save_path,memo)
			model_file='%s/%s.hdf5'%(opts.save_path,memo)
			o=open(loss_file,'wb')
			pickle.dump(loss,o)
			o.close()
			model.save_weights(model_file)
			
	
