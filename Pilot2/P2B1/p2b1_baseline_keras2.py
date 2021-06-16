import numpy as np
import sys
import os
import logging
try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3

TIMEOUT = 3600  # in sec; set this to -1 for no timeout
file_path = os.path.dirname(os.path.realpath(__file__))
# lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
# sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

from tensorflow.keras import backend as K

import p2b1
import candle

import p2b1_AE_models as AE_models

HOME = os.environ['HOME']

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_list(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def initialize_parameters(default_model='p2b1_default_model.txt'):

    # Build benchmark object
    p2b1Bmk = p2b1.BenchmarkP2B1(p2b1.file_path, default_model, 'keras',
                                 prog='p2b1_baseline', desc='Train Molecular Frame Autoencoder - Pilot 2 Benchmark 1')

    # Initialize parameters
    GP = candle.finalize_parameters(p2b1Bmk)
    # p2b1.logger.info('Params: {}'.format(gParameters))

    print('\nTraining parameters:')
    for key in sorted(GP):
        print("\t%s: %s" % (key, GP[key]))

    # print json.dumps(GP, indent=4, skipkeys=True, sort_keys=True)

    if GP['backend'] != 'theano' and GP['backend'] != 'tensorflow':
        sys.exit('Invalid backend selected: %s' % GP['backend'])

    os.environ['KERAS_BACKEND'] = GP['backend']
    reload(K)
    '''
    if GP['backend'] == 'theano':
        K.set_image_dim_ordering('th')
    elif GP['backend'] == 'tensorflow':
        K.set_image_dim_ordering('tf')
    '''
    K.set_image_data_format('channels_last')
# "th" format means that the convolutional kernels will have the shape (depth, input_depth, rows, cols)

# "tf" format means that the convolutional kernels will have the shape (rows, cols, input_depth, depth)
    print("Image data format: ", K.image_data_format())
#    print "Image ordering: ", K.image_dim_ordering()
    return GP


def run(GP):

    # set the seed
    if GP['rng_seed']:
        np.random.seed(GP['rng_seed'])
    else:
        np.random.seed(np.random.randint(10000))

    # Set paths
    if not os.path.isdir(GP['home_dir']):
        print('Keras home directory not set')
        sys.exit(0)
    sys.path.append(GP['home_dir'])

    # Setup loggin
    args = candle.ArgumentStruct(**GP)
#    set_seed(args.rng_seed)
#    ext = extension_from_parameters(args)
    candle.verify_path(args.save_path)
    prefix = args.save_path  # + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    candle.set_up_logger(logfile, logger, False)  # args.verbose
    logger.info('Params: {}'.format(GP))

    import p2b1 as hf
    reload(hf)

    # import keras_model_utils as KEU
    # reload(KEU)
    # reload(p2ck)
    # reload(p2ck.optimizers)
    # maps = hf.autoencoder_preprocess()

    # from keras.callbacks import LearningRateScheduler
    from tensorflow.keras import callbacks

#    GP=hf.ReadConfig(opts.config_file)
    batch_size = GP['batch_size']
    learning_rate = GP['learning_rate']
    kerasDefaults = candle.keras_default_config()

# #### Read Data ########
    import helper
    (data_files, fields) = p2b1.get_list_of_data_files(GP)
    # Read from local directoy
    # (data_files, fields) = helper.get_local_files('/p/gscratchr/brainusr/datasets/cancer/pilot2/3k_run16_10us.35fs-DPPC.20-DIPC.60-CHOL.20.dir/')
    # (data_files, fields) = helper.get_local_files('3k_run16', '/p/lscratchf/brainusr/datasets/cancer/pilot2/')

    # Define datagenerator
    # datagen = hf.ImageNoiseDataGenerator(corruption_level=GP['noise_factor'])

    # get data dimension ##
    num_samples = 0
    for f in data_files:

        # Seperate different arrays from the data
        (X, nbrs, resnums) = helper.get_data_arrays(f)

        num_samples += X.shape[0]

    (X, nbrs, resnums) = helper.get_data_arrays(data_files[0])
    print('\nData chunk shape: ', X.shape)

    molecular_hidden_layers = GP['molecular_num_hidden']
    if not molecular_hidden_layers:
        X_train = hf.get_data(X, case=GP['case'])
        input_dim = X_train.shape[1]
    else:
        # computing input dimension for outer AE
        input_dim = X.shape[1] * molecular_hidden_layers[-1]

    print('\nState AE input/output dimension: ', input_dim)

    # get data dimension for molecular autoencoder
    molecular_nbrs = np.int(GP['molecular_nbrs'])
    num_molecules = X.shape[1]
    num_beads = X.shape[2]

    if GP['nbr_type'] == 'relative':
        # relative x, y, z positions
        num_loc_features = 3
        loc_feat_vect = ['rel_x', 'rel_y', 'rel_z']
    elif GP['nbr_type'] == 'invariant':
        # relative distance and angle
        num_loc_features = 2
        loc_feat_vect = ['rel_dist', 'rel_angle']
    else:
        print('Invalid nbr_type!!')
        exit()

    if not GP['type_bool']:
        # only consider molecular location coordinates
        num_type_features = 0
        type_feat_vect = []
    else:
        num_type_features = 5
        type_feat_vect = list(fields.keys())[3:8]

    num_features = num_loc_features + num_type_features + num_beads
    dim = np.prod([num_beads, num_features, molecular_nbrs + 1])
    bead_kernel_size = num_features
    molecular_input_dim = dim
    mol_kernel_size = num_beads

    feature_vector = loc_feat_vect + type_feat_vect + list(fields.keys())[8:]

    print('\nMolecular AE input/output dimension: ', molecular_input_dim)

    print('\nData Format:\n[Frames (%s), Molecules (%s), Beads (%s), %s (%s)]' % (
        num_samples, num_molecules, num_beads, feature_vector, num_features))

# ## Define Model, Solver and Compile ##########
    print('\nDefine the model and compile')
    opt = candle.build_optimizer(GP['optimizer'], learning_rate, kerasDefaults)
    # model_type = 'mlp'
    # memo = '%s_%s' % (GP['base_memo'], model_type)

# ####### Define Molecular Model, Solver and Compile #########
    molecular_nonlinearity = GP['molecular_nonlinearity']

    len_molecular_hidden_layers = len(molecular_hidden_layers)
    conv_bool = GP['conv_bool']
    full_conv_bool = GP['full_conv_bool']
    if conv_bool:
        molecular_model, molecular_encoder = AE_models.conv_dense_mol_auto(bead_k_size=bead_kernel_size,
                                                                           mol_k_size=mol_kernel_size,
                                                                           weights_path=None,
                                                                           input_shape=(1, molecular_input_dim, 1),
                                                                           nonlinearity=molecular_nonlinearity,
                                                                           hidden_layers=molecular_hidden_layers,
                                                                           l2_reg=GP['l2_reg'],
                                                                           drop=float(GP['dropout']))
    elif full_conv_bool:
        molecular_model, molecular_encoder = AE_models.full_conv_mol_auto(bead_k_size=bead_kernel_size,
                                                                          mol_k_size=mol_kernel_size,
                                                                          weights_path=None,
                                                                          input_shape=(1, molecular_input_dim, 1),
                                                                          nonlinearity=molecular_nonlinearity,
                                                                          hidden_layers=molecular_hidden_layers,
                                                                          l2_reg=GP['l2_reg'],
                                                                          drop=float(GP['dropout']))

    else:
        molecular_model, molecular_encoder = AE_models.dense_auto(weights_path=None, input_shape=(molecular_input_dim,),
                                                                  nonlinearity=molecular_nonlinearity,
                                                                  hidden_layers=molecular_hidden_layers,
                                                                  l2_reg=GP['l2_reg'],
                                                                  drop=float(GP['dropout']))

    if GP['loss'] == 'mse':
        loss_func = 'mse'
    elif GP['loss'] == 'custom':
        loss_func = helper.combined_loss

    molecular_model.compile(optimizer=opt, loss=loss_func, metrics=['mean_squared_error', 'mean_absolute_error'])
    print('\nModel Summary: \n')
    molecular_model.summary()
    # #### set up callbacks and cooling for the molecular_model ##########
    drop = GP['dropout']
    mb_epochs = GP['epochs']
    initial_lrate = GP['learning_rate']
    epochs_drop = 1 + int(np.floor(mb_epochs / 3))

    def step_decay(epoch):
        global initial_lrate, epochs_drop, drop
        lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))
        return lrate

    # lr_scheduler = LearningRateScheduler(step_decay)
    history = callbacks.History()
    # callbacks=[history,lr_scheduler]

    history_logger = candle.LoggingCallback(logger.debug)
    candleRemoteMonitor = candle.CandleRemoteMonitor(params=GP)
    timeoutMonitor = candle.TerminateOnTimeOut(TIMEOUT)
    callbacks = [history, history_logger, candleRemoteMonitor, timeoutMonitor]
    # loss = 0.

# ### Save the Model to disk
    if GP['save_path'] is not None:
        save_path = GP['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = '.'

    model_json = molecular_model.to_json()
    with open(save_path + '/model.json', "w") as json_file:
        json_file.write(model_json)

    encoder_json = molecular_encoder.to_json()
    with open(save_path + '/encoder.json', "w") as json_file:
        json_file.write(encoder_json)

    print('Saved model to disk')

# ### Train the Model
    if GP['train_bool']:
        ct = hf.Candle_Molecular_Train(molecular_model, molecular_encoder, data_files, mb_epochs, callbacks,
                                       batch_size=batch_size, nbr_type=GP['nbr_type'], save_path=GP['save_path'],
                                       len_molecular_hidden_layers=len_molecular_hidden_layers,
                                       molecular_nbrs=molecular_nbrs,
                                       conv_bool=conv_bool,
                                       full_conv_bool=full_conv_bool,
                                       type_bool=GP['type_bool'],
                                       sampling_density=GP['sampling_density'])
        frame_loss, frame_mse = ct.train_ac()
    else:
        frame_mse = []
        frame_loss = []

    return frame_loss, frame_mse


def main():

    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:      # theano does not have this function
        pass
