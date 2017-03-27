import theano
import numpy as np
import scipy as sp
import pickle
import sys,os
import glob
import optparse
from keras.utils.data_utils import get_file
HOME=os.environ['HOME']
def parse_list(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

if __name__=="__main__":
### Hyperparameters and model save path
	parser=optparse.OptionParser()
	parser.add_option("--train", action="store_true",dest="train_bool",default=False,help="Invoke training")
	parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='.')
	parser.add_option("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
	parser.add_option("--config-file",help="Config File",dest="config_file",type=str,default='./config.ini')
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	parser.add_option("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
	(opts,args)=parser.parse_args()


	if not os.path.isdir(opts.home_dir):
		print ('Keras home directory not set')
		sys.exit(0)
	sys.path.append('home_dir')
	
	import candle_helper_functions as hf
	reload(hf)
        lib_path = os.path.abspath(os.path.join('..', 'common'))
        sys.path.append(lib_path)
	import keras_model_utils as KEU
	reload(KEU)
	maps=hf.autoencoder_preprocess()
	#from keras.Helpermodules.mnist_autoencoder_helper_functions import mnist_conv_deconv_simple, mnist_conv_deconv_complex,mnist_autoencoder_preprocess,generate_figure
	
	from keras.optimizers import SGD,RMSprop,Adam
	from keras.datasets import mnist
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras.regularizers import l2,WeightRegularizer
	from keras import callbacks
	from keras.layers.advanced_activations import ELU
	from keras.preprocessing.image import ImageDataGenerator

	GP=hf.ReadConfig(opts.config_file)
	batch_size = GP['batch_size']

##### Read Data ########
	print ('Reading Data...')
        data_file = get_file('p2_small_baseline.npy', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P2B1/p2_small_baseline.npy', md5_hash="a7769c9521f758c858a549965d04349d")
#	data_file='%s/Work/DataSets/CANDLE/sim-numpy.npy'%HOME ### can code to read at the terminal
	print 'Data File: %s' %data_file
	print 'Data Format: [Num Samples, Num Molecules, Num Atoms, Position]'
	
	X=np.load(data_file) ### Data is: Samples, Molecules, Atoms, x-pos,y-pos,z-pos
	if opts.case.upper()=='FULL':
		print 'Design autoencoder for data frame with coordinates for all beads'
		X_train=X.copy().reshape(X.shape[0],np.prod(X.shape[1:]))
	if opts.case.upper()=='CENTER':
		print 'design autoencoder for data frame with coordinates of the center-of-mass'
		X_train=X.mean(axis=2).reshape(X.shape[0],np.prod(X.mean(axis=2).shape[1:]))
	if opts.case.upper()=='CENTERZ':
		print 'design autoencoder for data frame with z-coordiate of the center-of-mass'
		X_train=X.mean(axis=2)[:,:,2].reshape(X.shape[0],np.prod(X.mean(axis=2)[:,:,2].shape[1:]))

	y_train=X_train.copy()
	input_dim=X_train.shape[1]
	mu, sigma = np.mean(X_train), np.std(X_train)
	mu=0.0;sigma=1.0
	X_train=maps.renormalize(X_train,mu,sigma)
	datagen=hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])  ## Add some corruption to input data ## idead for denoising auto encoder 
		
	print('X_train type and shape:', X_train.dtype, X_train.shape)
	print('X_train.min():', X_train.min())
	print('X_train.max():', X_train.max())

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
	
### Set up for Training and Validation
	total_epochs = GP['epochs']
	initial_lrate=GP['learning_rate']
	if GP['cool']:
		drop=0.5
	else:
		drop=1.0
	
	epochs_drop=1+int(np.floor(total_epochs/3))
		
	def step_decay(epoch):
		global initial_lrate,epochs_drop,drop
		lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
		return lrate
	lr_scheduler = LearningRateScheduler(step_decay)

#### Train the Model
	if opts.train_bool:
		history = callbacks.History()
		if opts.save_path !=None:
			model_file='%s/%s.hdf5'%(opts.save_path,memo)
			checkpointer=ModelCheckpoint(filepath=model_file, verbose=1)
			callbacks=[history,lr_scheduler,checkpointer]
		else:
			callbacks=[history,lr_scheduler]
		model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),\
			samples_per_epoch=X_train.shape[0],nb_epoch=total_epochs,callbacks=callbacks,verbose=1)
		loss_data={'train': history.history['loss']}
		if opts.save_path!=None:
			loss_file='%s/%s.pkl'%(opts.save_path,memo)
			o=open(loss_file,'wb')
			pickle.dump(loss_data,o)
			o.close()
	
	
