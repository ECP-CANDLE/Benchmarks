# RNN Generator
Based 99.98\% on the model from [1]

## Usage

The CANDLE-ized versions of the codes can simply be run without any command line arguments, with the default settings being read from the corresponding `default_model` file.
When needed, the CANDLE versions also use the `fetch_file` methods, which store the data in the top-level `Data/Examples` directory.
Any keywords in the `default_model` file can be overwritten with the appropriate command line argument.
The orginal codes and workflow below are preserved for comparison.
New package dependencies are now included in the top-level install instructions.

# CANDLE workflow

This will automatically download the models needed and run with the `autosave.model.pt` set in the `default_model` file.

```
python infer_rnngen_baseline_pytorch.py
```

# Original workflow

## Python dependencies

You need to set up a conda env. This will look different for different machines. 

```python
conda create -n myenv python=3.6
conda activate my env

conda install pytorch torchvision
conda install gensim matplotlib pandas tqdm 
conda install -c rdkit rdkit 
```

## Checkout pre-trained models
```
./fetch_models.sh
```

## Run inferences
1. Sample Compounds 
```shell
python infer.py -i mosesrun/ --logdir mosesrun/ -o samples.txt -n 10000 -vr 
```
This will use the model from chemblrun to create 10,000 samples and validate the samples. 

2. Pilot1 Drug based models
```shell
python infer.py -i mosesrun/ --logdir pilot1/ -o p1_good.txt -n 10000 -vr --model ft_goodperforming_model.pt

python infer.py -i mosesrun/ --logdir pilot1/ -o p1_poor.txt -n 10000 -vr --model ft_poorperforming_model.pt
```


# Refereces:
1. Gupta, A., Müller, A., Huisman, B., Fuchs, J., Schneider, P., Schneider, G. (2018). Generative Recurrent Networks for De Novo Drug Design Molecular Informatics  37(1-2) https://dx.doi.org/10.1002/minf.201700111
2. Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Nikolenko, S., Aspuru-Guzik, A., Zhavoronkov, A. (2018). Molecular Sets (MOSES): A Benchmarking Platform for Molecular Generation Models https://arxiv.org/abs/1811.12823

