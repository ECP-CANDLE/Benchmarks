import theano
import numpy as np
import scipy as sp
import pickle
import sys,os
import glob
import optparse
import matplotlib
matplotlib.use('TKAgg')
import pylab as py
py.ion()
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
	parser.add_option("--batch-size",help="batch size",dest="batch_size",type=int,default=1)
	parser.add_option("--look-back",help="look back time window",dest="look_back",type=int,default=1)
	parser.add_option("--home-dir",help="Home Directory",dest="home_dir",type=str,default='/Users/talathi1/Work/Python/Git_Folder/caffe-tools/keras')
	parser.add_option("--save-dir",help="Save Directory",dest="save_path",type=str,default=None)
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	parser.add_option("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
	(opts,args)=parser.parse_args()

	## set the seed
	if opts.seed:
		np.random.seed(7)
	else:
		np.random.seed(np.random.randint(10000))

	## Set paths
	if not os.path.isdir(opts.home_dir):
		print ('Keras home directory not set')
		sys.exit(0)
	sys.path.append('home_dir')
	
	import candle_helper_functions as hf
	reload(hf)
	maps=hf.autoencoder_preprocess()
	
	## Import keras modules
	from keras.optimizers import SGD,RMSprop,Adam
	from keras.datasets import mnist
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras.regularizers import l2,WeightRegularizer
	from keras import callbacks
	from keras.layers.advanced_activations import ELU
	from keras.preprocessing.image import ImageDataGenerator

	batch_size = opts.batch_size
##### Read Data ########
	print ('Reading Data...')
	data_file='%s/Research/DeepLearning/ECP CANDLE/Benchmarks/Benchmarks.git/P2B2/sim-numpy.npy'%HOME ### can code to read at the terminal
	print 'Data File: %s' %data_file
	print 'Data Format: [Num Samples, Num Molecules, Num Atoms, Position]'
	
	X=np.load(data_file) ### Data is: Samples, Molecules, Atoms, x-pos,y-pos,z-pos
	## Take center of mass for atoms:
	X_A=X.mean(axis=2) ## Data is: Samples, Molecules, x-pos,y-pos,z-pos
	data=X_A[0:-1,:,2] ## only consider z-dimension
	X_train,y_train=hf.create_dataset(data,opts.look_back,look_forward=1) ## convert data to a sequence 
	temporal_dim=X_train.shape[1]
	input_dim=X_train.shape[2]
	subset_sample_weight=np.ones((X_train.shape[0],1))
	sample_weight=np.zeros((X_train.shape[0],opts.look_back))
	sample_weight[:,0:1]=subset_sample_weight

	print('X_train type and shape:', X_train.dtype, X_train.shape)
	print('X_train.min():', X_train.min())
	print('X_train.max():', X_train.max())

### Define Model, Solver and Compile ##########
	print ('Define the model and compile')
	opt=Adam(lr=opts.learning_rate)

	print ('using mlp network')
	model_type='mlp'
	hidden_layers=[512,256,128,64,32,16]
	recurrent_layers=[16,16,16]
	## Model is a Autoencoder-RNN network
	model=hf.rnn_dense_auto(weights_path=None,T=temporal_dim,D=input_dim,nonlinearity='relu',hidden_layers=hidden_layers,recurrent_layers=recurrent_layers)
		
	memo='%s_%s_%0.5f'%(opts.base_memo,model_type,opts.learning_rate)

	print 'Autoencoder Regression problem'
	model.compile(optimizer=opt, loss='mean_squared_error',sample_weight_mode="temporal")
	model.summary()	## print model summary in details
	#sys.exit(0)

#### Print Compact Model Configuration ###########
	# num_layers=len(model.layers)
	# print '*'*10,'Model Configuration','*'*10
	# for i  in range(len(model.layers)):	
	# 	print i,': ',model.layers[i].name, ':', model.layers[i].output_shape[:]

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
		model.fit(X_train, y_train, batch_size=batch_size,shuffle=False,nb_epoch=total_epochs,callbacks=callbacks,verbose=1,sample_weight=sample_weight)
		
		loss_data={'train': history.history['loss']}
		if opts.save_path!=None:
			loss_file='%s/%s.pkl'%(opts.save_path,memo)
			o=open(loss_file,'wb')
			pickle.dump(loss_data,o)
			o.close()

		## Generate model forecast figure  
		x=X_train[0:1]
		xmod=x.reshape(x.shape[1],x.shape[2])
		yf=hf.generate_timedistributed_forecast(model,x,X_train.shape[0]+opts.look_back)
		yt=yt=y_train[:,0,:]
		ytn=np.vstack([xmod,yt])
		py.figure();py.plot(ytn.mean(axis=1))
		py.hold('on');py.plot(yf.mean(axis=1))



	
	
