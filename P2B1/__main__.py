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
	parser.add_option("--learning-rate",help="learning rate",dest="learning_rate",type=float,default=0.1)
	parser.add_option("--noise-factor",help="noise",dest="noise_factor",type=float,default=0.0)
	parser.add_option("--cool",action="store_true",dest="cool",default=False,help="Cool Learning Rate")
	parser.add_option("--epochs",help="epochs",dest="epochs",type=int,default=1)
	parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='/Users/talathi1/Work/Python/Git_Folder/caffe-tools/keras')
	parser.add_option("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	(opts,args)=parser.parse_args()

## Example of training command:
#python mnist_conv_autoencoder.py -C --save-dir /Users/talathi1/Work/Python/Models/Test_Models --memo test --epochs 1 --data-path /Users/talathi1/Work/DataSets/mnist.pkl.gz --learning-rate 0.01 --train --classify --unpool_type 3

	if not os.path.isdir(opts.home_dir):
		print ('Keras home directory not set')
		sys.exit(0)
	sys.path.append('home_dir')
	
	import candle_helper_functions as hf
	reload(hf)
	maps=hf.autoencoder_preprocess()
	#from keras.Helpermodules.mnist_autoencoder_helper_functions import mnist_conv_deconv_simple, mnist_conv_deconv_complex,mnist_autoencoder_preprocess,generate_figure
	
	from keras.optimizers import SGD,RMSprop
	from keras.datasets import mnist
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras.regularizers import l2,WeightRegularizer
	from keras import callbacks
	from keras.layers.advanced_activations import ELU
	from keras.preprocessing.image import ImageDataGenerator

	batch_size = 16
##### Read Data ########
	print ('Reading Data...')
        data_file = get_file('p2_small_baseline.npy', origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/P2B1/p2_small_baseline.npy')
	print 'Data File: %s' %data_file
	print 'Data Format: [Num Samples, Num Molecules, Num Atoms, Position]'
	
	X=np.load(data_file) ### Data is: Samples, Molecules, Atoms, x-pos,y-pos,z-pos
	## Take center of mass for atoms:
	X_A=X.mean(axis=2) ## Data is: Samples, Molecules, x-pos,y-pos,z-pos
	#X_train=X_A.reshape(X_A.shape[0],X_A.shape[1]*X_A.shape[2])
	X_train=X_A[:,:,2] ## only consider z-dimension
	y_train=X_train.copy()
	input_dim=X_train.shape[1]
	mu, sigma = np.mean(X_train), np.std(X_train)
	mu=0.0;sigma=1.0
	X_train=maps.renormalize(X_train,mu,sigma)
	datagen=hf.ImageNoiseDataGenerator(corruption_level=opts.noise_factor)  ## Add some corruption to input data ## idead for denoising auto encoder 
		
	print('X_train type and shape:', X_train.dtype, X_train.shape)
	print('X_train.min():', X_train.min())
	print('X_train.max():', X_train.max())

### Define Model, Solver and Compile ##########
	print ('Define the model and compile')
	opt = SGD(lr=opts.learning_rate, decay=0.0, momentum=0.975, nesterov=True)
	
	print ('using mlp network')
	model_type='mlp'
	hidden_layers=[512,256,128,64,32,16]
	model=hf.dense_auto(weights_path=opts.weight_path,input_shape=(input_dim,),nonlinearity='elu',hidden_layers=hidden_layers)
		
	memo='%s_%s_%0.5f'%(opts.base_memo,model_type,opts.learning_rate)

	print 'Autoencoder Regression problem'
	model.compile(optimizer='adadelta', loss='mean_squared_error')

#### Print Model Configuration ###########
	num_layers=len(model.layers)
	print '*'*10,'Model Configuration','*'*10
	for i  in range(len(model.layers)):	
		print i,': ',model.layers[i].name, ':', model.layers[i].output_shape[:]

### Set up for Training and Validation
	total_epochs = opts.epochs
	initial_lrate=opts.learning_rate
	if opts.cool:
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
	
	
