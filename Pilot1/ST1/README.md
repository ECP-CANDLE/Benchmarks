# Simple transformers for classification and regression using SMILE string input

## Introduction

The ST1 benchmark represent different versions of a simple transformer that can either do classification or regression. We chose the transformer architecture to see if we could train directly on SMILE strings. This benchmark brings novel capability to the suite of Pilot1 benchmarks in two ways. First, featurization of a small molecule is simply a tokenization of the SMILES string. The second novel aspect to the set of Pilot1 benchmarks is that the model is based on the Transformer architecture, albeit this benchmark is a simpler version of the large Transformer models that train on billions and greater parameters.

Both the original code and the CANDLE versions are available. The original examples are retained and can be run as noted below. The CANDLE versions make use of the common network design in `smiles_transformer.py`, and implement the models in `sct_baseline_keras2.py` and `srt_baseline_keras2.py`, for classification and regression, respectively.

The example classification problem takes as input SMILE strings and trains a model to predict whether or not a compound is 'drug-like' based on Lipinski criteria. The example regression problem takes as input SMILE strings and trains a model to predict the molecular weight. Data are freely downloadable and automatically downloaded by the CANDLE versions.

For the CANDLE versions, all the relevant arguments are contained in the respective default model files. All variables can be overwritten from the command line. The datasets will be automatically downloaded and stored in the `../../Data/Pilot1 directory`. The respective default model files and commands to invoke the classifier and regressor are:

Additional developments to ST1 are three implementations designed for performing regression on SMILES to predict binding affinity to macromolecular targets.
These implementations are:

(1) ST1 original: The original ST1 code initially trained to predict molecular weight now trained on binding affinity measurements.

(2) ST1-horovod: ST1 original with additional functionality allowing for distributed training with horovod.

(3) ST1 with SPE tokenizer: ST1 model trained on binding affinity measurements that featurizes SMILES strings using a special byte-pair encoder known as SMILES-pair encoder (https://doi.org/10.1021/acs.jcim.0c01127). We show this implementation improves accuracy of the model and reduces the overall model size (thus improving inference speed).

```
class_default_model.txt
python sct_baseline_keras2.py
```

and

```
regress_default_model.txt
python srt_baseline_keras2.py
```

## Running the original versions

The original code demonstrating a simple transformer regressor and a simple transformer classifier are available as

```
smiles_regress_transformer.py
```

and

```
smiles_class_transformer.py
```

The example data sets are the same as for the CANDLE versions, and allow one to predict whether a small molecule is "drug-like" based on Lipinski criteria (classification problem), or predict the molecular weight (regression) from a SMILE string as input.
The example data sets are downloadable using the information in the `regress_default_model.txt` or `class_default_model.txt` files.
These data files must be downloaded manually and specified on the command line for execution of the original versions.

```
# for regression
train_data = https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Examples/xform-smiles-data/chm.weight.trn.csv
val_data = https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Examples/xform-smiles-data/chm.weight.val.csv


# for classification
train_data = https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Examples/xform-smiles-data/chm.lipinski.trn.csv
val_data = https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Examples/xform-smiles-data/chm.lipinski.val.csv
```

To run the models

```
CUDA_VISIBLE_DEVICES=1 python smiles_class_transformer.py --in_train chm.lipinski.trn.csv --in_vali chm.lipinski.val.csv --ep 25
```

or

```
CUDA_VISIBLE_DEVICES=0 python smiles_regress_transformer.py --in_train chm.weight.trn.csv --in_vali chm.weight.val.csv --ep 25
```

The model with the best validation loss is saved in the .h5 dumps. Log files contain the trace. Regression output should look something like this.

```
Epoch 1/25
2022-03-21 12:53:11.402337: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
46875/46875 [==============================] - 4441s 95ms/step - loss: 11922.3690 - mae: 77.4828 - r2: 0.3369 - val_loss: 1460.3314 - val_mae: 21.4797 - val_r2: 0.9164

Epoch 00001: val_loss improved from inf to 1460.33142, saving model to smile_regress.autosave.model.h5
Epoch 2/25
46875/46875 [==============================] - 4431s 95ms/step - loss: 5460.8232 - mae: 55.1082 - r2: 0.6845 - val_loss: 1451.3109 - val_mae: 27.5647 - val_r2: 0.9163

Epoch 00002: val_loss improved from 1460.33142 to 1451.31091, saving model to smile_regress.autosave.model.h5
Epoch 3/25
46875/46875 [==============================] - 4428s 94ms/step - loss: 5007.1718 - mae: 52.5552 - r2: 0.7112 - val_loss: 1717.9198 - val_mae: 31.0688 - val_r2: 0.9004

Epoch 00003: val_loss did not improve from 1451.31091
Epoch 4/25
46875/46875 [==============================] - 4419s 94ms/step - loss: 4844.3624 - mae: 51.5771 - r2: 0.7206 - val_loss: 1854.3645 - val_mae: 35.4912 - val_r2: 0.8908

Epoch 00004: val_loss did not improve from 1451.31091
Epoch 5/25
46875/46875 [==============================] - 4715s 101ms/step - loss: 4754.6214 - mae: 51.0409 - r2: 0.7277 - val_loss: 1404.8254 - val_mae: 25.8863 - val_r2: 0.9200

Epoch 00005: val_loss improved from 1451.31091 to 1404.82544, saving model to smile_regress.autosave.model.h5
Epoch 6/25
46875/46875 [==============================] - 4474s 95ms/step - loss: 4679.2172 - mae: 50.7018 - r2: 0.7314 - val_loss: 1353.6165 - val_mae: 23.1900 - val_r2: 0.9252

Epoch 00006: val_loss improved from 1404.82544 to 1353.61646, saving model to smile_regress.autosave.model.h5
Epoch 7/25
46875/46875 [==============================] - 4421s 94ms/step - loss: 4585.5881 - mae: 50.3534 - r2: 0.7359 - val_loss: 1312.0532 - val_mae: 27.0301 - val_r2: 0.9234

Epoch 00007: val_loss improved from 1353.61646 to 1312.05322, saving model to smile_regress.autosave.model.h5
Epoch 8/25
46875/46875 [==============================] - 4420s 94ms/step - loss: 4587.0872 - mae: 50.2153 - r2: 0.7348 - val_loss: 1064.2247 - val_mae: 19.8623 - val_r2: 0.9399

Epoch 00008: val_loss improved from 1312.05322 to 1064.22473, saving model to smile_regress.autosave.model.h5
Epoch 9/25
46875/46875 [==============================] - 4423s 94ms/step - loss: 4587.0479 - mae: 50.0801 - r2: 0.7369 - val_loss: 1592.4563 - val_mae: 29.7814 - val_r2: 0.9085

Epoch 00009: val_loss did not improve from 1064.22473
Epoch 10/25
46875/46875 [==============================] - 4425s 94ms/step - loss: 4558.5997 - mae: 49.9495 - r2: 0.7383 - val_loss: 1683.6605 - val_mae: 33.6503 - val_r2: 0.9023

Epoch 00010: val_loss did not improve from 1064.22473
Epoch 11/25
46875/46875 [==============================] - 4422s 94ms/step - loss: 4546.4993 - mae: 49.8544 - r2: 0.7401 - val_loss: 950.7401 - val_mae: 17.9388 - val_r2: 0.9465

Epoch 00011: val_loss improved from 1064.22473 to 950.74005, saving model to smile_regress.autosave.model.h5
Epoch 12/25
46875/46875 [==============================] - 4425s 94ms/step - loss: 4442.8416 - mae: 49.5132 - r2: 0.7436 - val_loss: 1060.0773 - val_mae: 18.5870 - val_r2: 0.9397

Epoch 00012: val_loss did not improve from 950.74005
Epoch 13/25
46875/46875 [==============================] - 4429s 94ms/step - loss: 4481.1873 - mae: 49.5325 - r2: 0.7434 - val_loss: 878.7765 - val_mae: 14.4166 - val_r2: 0.9515

Epoch 00013: val_loss improved from 950.74005 to 878.77649, saving model to smile_regress.autosave.model.h5
Epoch 14/25
46875/46875 [==============================] - 4428s 94ms/step - loss: 4474.0927 - mae: 49.4286 - r2: 0.7445 - val_loss: 1116.5386 - val_mae: 23.1262 - val_r2: 0.9360

Epoch 00014: val_loss did not improve from 878.77649
Epoch 15/25
46875/46875 [==============================] - 4422s 94ms/step - loss: 4426.6833 - mae: 49.2094 - r2: 0.7446 - val_loss: 883.4428 - val_mae: 13.1046 - val_r2: 0.9507

Epoch 00015: val_loss did not improve from 878.77649
Epoch 16/25
46875/46875 [==============================] - 4416s 94ms/step - loss: 4386.3840 - mae: 49.0872 - r2: 0.7476 - val_loss: 981.3953 - val_mae: 14.7772 - val_r2: 0.9451

Epoch 00016: val_loss did not improve from 878.77649
Epoch 17/25
46875/46875 [==============================] - 4423s 94ms/step - loss: 4384.2891 - mae: 49.0281 - r2: 0.7464 - val_loss: 887.1695 - val_mae: 14.6686 - val_r2: 0.9506

Epoch 00017: val_loss did not improve from 878.77649
Epoch 18/25
46875/46875 [==============================] - 4417s 94ms/step - loss: 4398.2363 - mae: 48.9583 - r2: 0.7487 - val_loss: 849.0564 - val_mae: 12.2192 - val_r2: 0.9528

Epoch 00018: val_loss improved from 878.77649 to 849.05640, saving model to smile_regress.autosave.model.h5
Epoch 19/25
46875/46875 [==============================] - 4422s 94ms/step - loss: 4350.2122 - mae: 48.8221 - r2: 0.7505 - val_loss: 847.2310 - val_mae: 13.4015 - val_r2: 0.9533

Epoch 00019: val_loss improved from 849.05640 to 847.23096, saving model to smile_regress.autosave.model.h5
Epoch 20/25
46875/46875 [==============================] - 4428s 94ms/step - loss: 4346.0122 - mae: 48.7704 - r2: 0.7513 - val_loss: 964.1797 - val_mae: 17.7863 - val_r2: 0.9453

Epoch 00020: val_loss did not improve from 847.23096
Epoch 21/25
46875/46875 [==============================] - 4424s 94ms/step - loss: 4293.9141 - mae: 48.5882 - r2: 0.7521 - val_loss: 800.8525 - val_mae: 12.8857 - val_r2: 0.9556

Epoch 00021: val_loss improved from 847.23096 to 800.85254, saving model to smile_regress.autosave.model.h5
Epoch 22/25
46875/46875 [==============================] - 4427s 94ms/step - loss: 4323.8214 - mae: 48.5665 - r2: 0.7524 - val_loss: 835.3901 - val_mae: 14.5708 - val_r2: 0.9534

Epoch 00022: val_loss did not improve from 800.85254
Epoch 23/25
46875/46875 [==============================] - 4429s 94ms/step - loss: 4311.1271 - mae: 48.5622 - r2: 0.7528 - val_loss: 820.6389 - val_mae: 14.3753 - val_r2: 0.9547

Epoch 00023: val_loss did not improve from 800.85254
Epoch 24/25
46875/46875 [==============================] - 4427s 94ms/step - loss: 4286.2930 - mae: 48.3628 - r2: 0.7548 - val_loss: 815.2863 - val_mae: 12.7142 - val_r2: 0.9549

Epoch 00024: val_loss did not improve from 800.85254
Epoch 25/25
46875/46875 [==============================] - 4422s 94ms/step - loss: 4259.8291 - mae: 48.3112 - r2: 0.7556 - val_loss: 813.1475 - val_mae: 12.2975 - val_r2: 0.9564

Epoch 00025: val_loss did not improve from 800.85254
```

## Example classification problem metrics

CHEMBL -- 1.5M training examples
Predicting Lipinski criteria for drug likeness (1/0)
Validation 100K samples non-overlapping
Classification validation accuracy is about 91% after 10-20 epochs

## Example regression problem metrics

CHEMBL -- 1.5M training examples (shuffled and resampled so not same 1.5M as classification)
Predicting molecular Weight validation
Is also 100K samples non-overlapping.
Regression problem achieves R^2 about .95 after ~20 epochs.
