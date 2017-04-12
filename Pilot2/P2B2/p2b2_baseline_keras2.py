import theano
import numpy as np
import scipy as sp
import pickle
import sys,os
import glob
import optparse
import matplotlib
from sklearn.preprocessing import MinMaxScaler
matplotlib.use('TKAgg')
import pylab as py
py.ion()
lib_path = os.path.abspath(os.path.join('..', '..', 'common'))
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
	parser.add_option("--config-file",help="Config File",dest="config_file",type=str,default='./p2b2_small_model.txt')
	parser.add_option("--model-file",help="Trained Model Pickle File",dest="weight_path",type=str,default=None)
	parser.add_option("--memo",help="Memo",dest="base_memo",type=str,default=None)
	parser.add_option("--seed", action="store_true",dest="seed",default=False,help="Random Seed")
	parser.add_option("--case",help="[Full, Center, CenterZ]",dest="case",type=str,default='CenterZ')
	parser.add_option("--fig", action="store_true",dest="fig_bool",default=False,help="Generate Prediction Figure")
	parser.add_option("--data-set",help="[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]",dest="set_sel",
		type=str,default="3k_Disordered")
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
	sys.path.append(opts.home_dir)
	
	import p2b2 as hf
	reload(hf)
	
	file_path = os.path.dirname(os.path.realpath(__file__))
	lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
	sys.path.append(lib_path)
	
	import pilot2_datasets as p2
	reload(p2)
	maps=hf.autoencoder_preprocess()
	
	GP=hf.ReadConfig(opts.config_file)
	print GP

	## Import keras modules
	from keras.optimizers import SGD,RMSprop,Adam
	from keras.datasets import mnist
	from keras.callbacks import LearningRateScheduler,ModelCheckpoint
	from keras import callbacks
	from keras.layers.advanced_activations import ELU
	from keras.preprocessing.image import ImageDataGenerator

	batch_size = GP['batch_size']
##### Read Data ########
	print ('Reading Data...')
	datagen=hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])
        data_set=p2.data_sets[opts.set_sel][0]
        data_hash=p2.data_sets[opts.set_sel][1]
	print ('Reading Data Files... %s->%s' % (opts.set_sel, data_set))
        data_file = get_file(data_set, origin='http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot2/'+data_set+'.tar.gz', untar=True, md5_hash=data_hash)
        data_dir = os.path.join(os.path.dirname(data_file), data_set)
	data_files=glob.glob('%s/*.npy'%data_dir) 

	X=np.load(data_files[0])
	data=hf.get_data(X,case=opts.case)
	X_train,y_train=hf.create_dataset(data,GP['look_back'],look_forward=GP['look_forward']) ## convert data to a sequence 
	temporal_dim=X_train.shape[1]
	input_dim=X_train.shape[2]
	
	print('X_train type and shape:', X_train.dtype, X_train.shape)
	print('X_train.min():', X_train.min())
	print('X_train.max():', X_train.max())
	
### Define Model, Solver and Compile ##########
	print ('Define the model and compile')
	opt=Adam(lr=GP['learning_rate'])

	print ('using mlp network')
	model_type='mlp'
	hidden_layers=GP['num_hidden']
	if len(hidden_layers)==0:
		hidden_layers=None
	recurrent_layers=GP['num_recurrent']
	
	## Model is a Autoencoder-RNN network
	model=hf.rnn_dense_auto(weights_path=None,T=temporal_dim,D=input_dim,nonlinearity='relu',hidden_layers=hidden_layers,recurrent_layers=recurrent_layers)
		
	memo='%s_%s'%(opts.base_memo,model_type)

	print 'Autoencoder Regression problem'
	model.compile(optimizer=opt, loss='mean_squared_error',sample_weight_mode="temporal")
	model.summary()	## print model summary in details

#### Train the Model
	if opts.train_bool:
		if not str2bool(GP['cool']):
			effec_epochs=GP['epochs']
			ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=opts.case,look_back=GP['look_back'],look_forward=GP['look_forward'])
			loss=ct.train_ac()
		else:
			effec_epochs=GP['epochs']//3
			if effec_epochs==0:
				effec_epochs=1
			ct=hf.Candle_Train(datagen,model,data_files,effec_epochs,case=opts.case,look_back=GP['look_back'],look_forward=GP['look_forward'])
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
			o=open(loss_file,'wb')
			pickle.dump(loss,o)
			o.close()

		## Generate model forecast figure  
		if opts.fig_bool:
			x=X_train[0:1]
			xmod=x.reshape(x.shape[1],x.shape[2])
			yf=hf.generate_timedistributed_forecast(model,x,X_train.shape[0])
			yt=yt=y_train[:,0,:]
			ytn=np.vstack([xmod,yt])
			py.figure();py.plot(ytn.mean(axis=1))
			py.hold('on');py.plot(yf.mean(axis=1))



	
	
