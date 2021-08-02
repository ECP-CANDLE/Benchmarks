from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

import unet
import candle


def initialize_parameters():
    unet_common = unet.UNET(
        unet.file_path,
        'unet_params.txt',
        'keras',
        prog='unet_example',
        desc='UNET example'
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(unet_common)
    return gParameters


def run(gParameters):

    # load data
    x_train, y_train = unet.load_data()

    # example has 420 x 580
    model = unet.build_model(420, 580, gParameters['activation'], gParameters['kernel_initializer'])

    model.summary()
    model.compile(optimizer=gParameters['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

    model_chkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train,
                        batch_size=gParameters['batch_size'],
                        epochs=gParameters['epochs'],
                        verbose=1,
                        validation_split=0.3,
                        shuffle=True,
                        callbacks=[model_chkpoint]
                        )

    return history


def main():
    gParameters = initialize_parameters()
    run(gParameters)


if __name__ == '__main__':
    main()
    try:
        K.clear_session()
    except AttributeError:
        pass
