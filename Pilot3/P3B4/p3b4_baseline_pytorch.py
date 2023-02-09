
import numpy
import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import intel_extension_for_pytorch as ipex
from torch.utils.data import Dataset, DataLoader, Subset

# import candle
from dataloaders import PathReports
from mthisan import MTHiSAN
from training import *
import p3b4 as bmk
import candle

debug = True
logger_level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

def initialize_parameters(default_model='p3b4_default_model.txt'):
    """ Initialize the parameters for the P3B5 benchmark """

    # Build benchmark object
    p3b4Bmk = bmk.BenchmarkP3B4(bmk.file_path, default_model, 'keras',
                                prog='p3b4_baseline',
                                desc='Hierarchical Self-Attention Network for \
                                data extraction - Pilot 3 Benchmark 4')

    # Initialize parameters
    gParameters = candle.finalize_parameters(p3b4Bmk)

    return gParameters


def fetch_data(gParameters):
    """ Downloads and decompresses the data if not locally available.
    """

    path = gParameters['data_url']
    fpath = candle.fetch_file(
        path + gParameters['train_data'], 'Pilot3', unpack=True)

    return fpath


def run(gParameters):

    fpath = fetch_data(gParameters)
    learning_rate = gParameters['learning_rate']
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    dropout = gParameters['dropout']
    embed_train = gParameters['embed_train']
    wv_len = gParameters['wv_len']
    attention_size = gParameters['attention_size']
    attention_heads = gParameters['attention_heads']
    attention_dim_per_head = 64
    max_words = gParameters['max_words']
    max_lines = gParameters['max_lines']
    max_len = gParameters['max_len']
    tasks = gParameters['task_names']
    patience = gParameters['patience']

    train_x = np.load(fpath + '/train_X.npy')
    train_y = np.load(fpath + '/train_Y.npy')
    test_x = np.load(fpath + '/test_X.npy')
    test_y = np.load(fpath + '/test_Y.npy')

    np_X_train = train_x
    df_Y_train = pd.DataFrame(train_y, columns = tasks)
    np_X_val = test_x
    df_Y_val = pd.DataFrame(test_y, columns = tasks)
    np_X_test = test_x
    df_Y_test = pd.DataFrame(test_y, columns = tasks)

    # label encoder
    labelencoders = {}
    num_classes = []

    for task in tasks:
        le = LabelEncoder()
        codes = np.concatenate( ( df_Y_train[ task ], df_Y_val[ task ] ) )
        le.fit( codes )
        labelencoders[ task ] = le
        num_classes.append( len( le.classes_ ) )
    logger.info(labelencoders)
    logger.info(num_classes)

    np_X_train = np.flip( np_X_train, axis= 1 )
    np_X_val = np.flip( np_X_val, axis= 1 )
    np_X_test = np.flip( np_X_test, axis= 1 )

    # dataset objects
    train_dataset = PathReports(
        np_X= np_X_train,
        df_Y= df_Y_train,
        tasks= tasks,
        label_encoders= labelencoders,
        max_len= max_len,
    )

    val_dataset = PathReports(
        np_X= np_X_val,
        df_Y= df_Y_val,
        tasks= tasks,
        label_encoders= labelencoders,
        max_len= max_len,
    )

    test_dataset = PathReports(
        np_X= np_X_test,
        df_Y= df_Y_test,
        tasks= tasks,
        label_encoders= labelencoders,
        max_len= max_len,
    )

    logger.info( train_dataset )

    # data loader
    train_loader = DataLoader( train_dataset, batch_size= batch_size, shuffle= True )
    val_loader = DataLoader( val_dataset, batch_size= batch_size, shuffle= False )
    test_loader = DataLoader( test_dataset, batch_size= batch_size, shuffle= False )

    # model
    embedding_matrix = np.random.uniform( low = 0, high= 1, size= ( 200000, 300 ) )
    model = MTHiSAN(embedding_matrix,
            num_classes,
            max_words,
            max_lines,
            att_dim_per_head=attention_dim_per_head,
            att_heads=attention_heads,
            att_dropout=dropout
            )

    # training
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, ( 0.9, 0.99 ) )
    logger.info( 'training hisan using %i %s devices' % ( torch.xpu.device_count(), device ) )

    if not os.path.exists( 'checkpoints' ):
        os.makedirs( 'checkpoints' )

    model_filename = 'checkpoints/' + '.'.join( ['hisan', str( 0 ), 'model' ] )

    train( model= model,
        optimizer= optimizer,
        tasks= tasks,
        train_loader= train_loader,
        val_loader= val_loader,
        epochs= epochs,
        patience= patience,
        savepath= model_filename,
        )

    # load best model
    logger.info('Testing best saved model')
    model.load_state_dict(torch.load( model_filename ) )

    # test accuracy
    base_scores, abs_scores = score(
        model,
        tasks,
        test_loader,
    )

    # arrays to store all values including ntask
    micros = base_scores['micros']
    macros = base_scores['macros']
    val_loss = base_scores['val_loss']

    logger.info( 'Task, Micro, Macro' )
    for t, task in enumerate(tasks):
        print_stats( task, micros[ t ], macros[ t ] )


def main():

    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == '__main__':
    main()

