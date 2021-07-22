## Usage

The CANDLE-ized versions of the codes can simply be run without any command line arguments, with the default settings being read from the corresponding `default_model` file.
When needed, the CANDLE versions also use the `fetch_file` methods, which store the data in the top-level `Data/Examples` directory.
Any keywords in the `default_model` file can be overwritten with the appropriate command line argument.
The orginal codes and workflow below are preserved for comparison.
New package dependencies are now included in the top-level install instructions.

# CANDLE workflow

```
python image_vae_baseline_pytorch.py
python sample_baseline_pytorch.py
```

# Image VAE

2D-Images are a relatively unexplored representation for molecular learning tasks. We create a molecular generator and embedding based on 2D-depictions of molecules. We use a variational autoencoder (VAE) to encode 2D-images of molecules to a latent space of 512, and with a gaussian prior sample the space and decode directly to images. A modified ResNet is used to encode molecular depictions to a latent space. A decoder is created by performing the inverse operations of ResNet (i.e. run blocks in reverse order replacing convolutional layers with deconvolution layers (transpose convolution). One can embed molecules in this space by use of only the decoder, or by generating random gaussian noise one can decode a latent vector into a molecular image. In the latent space, generation can also be steered through interpolation or epsilon-sampling. VAEs are prone to mode collapse and exploding gradients. Mode collapse occurs when enforcing a normal prior on the latent space causes any learning to “collapse” and the model ceases to learn. Exploding gradients occurs when the gradients become so large that the optimization routine becomes unstable and again learning ceases to occur. To mode collapse, we use KL-divergence loss annealing. KL-divergence loss annealing slowly ramps up the weight of the normal prior in latent space as the model learns to reconstruct the encoded images better. This essentially enforces that the decoder and encoder learn at similar rates. The avoid exploding gradients, we use gradient clipping which limits the magnitude of any particular optimization step. This enforces a slow and gradual learning process.

## Requirement
```
 conda install -c rdkit rdkit
 conda install pytorch torchvision
 conda install scikit-learn

 pip install cairosvg
 ```

## Training

```
$ git clone https://github.com/molecularsets/moses
# python main.py -w 32 -b 1024

loading data...
                                   SMILES  SPLIT
0  CCCS(=O)c1ccc2[nH]c(=NC(=O)OC)[nH]c2c1  train
1    CC(C)(C)C(=O)C(Oc1ccc(Cl)cc1)n1ccnc1  train
2     Cc1c(Cl)cccc1Nc1ncccc1C(=O)OCC(O)CO  train
3        Cn1cnc2c1c(=O)n(CC(O)CO)c(=O)n2C  train
4          CC1Oc2ccc(Cl)cc2N(CC(O)CO)C1=O  train
                                        SMILES SPLIT
0       CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1  test
1  COc1ccc(-c2cc(=O)c3c(O)c(OC)c(OC)cc3o2)cc1O  test
2      CCOC(=O)c1ncn2c1CN(C)C(=O)c1cc(F)ccc1-2  test
3                Clc1ccccc1-c1nc(-c2ccncc2)no1  test
4      CC(C)(Oc1ccc(Cl)cc1)C(=O)OCc1cccc(CO)n1  test
Done.

LR: 0.0005
Let's use 8 GPUs!
Current learning rate is: 0.0005
Epoch 1: batch_size 1024
Train Epoch: 1 [0/198082 (0%)]	Loss: 47978.167969 2021-03-30 10:59:14.397963
Train Epoch: 1 [25600/198082 (13%)]	Loss: 19583.210355 2021-03-30 10:59:50.408786
Train Epoch: 1 [51200/198082 (26%)]	Loss: 12336.477529 2021-03-30 11:00:26.037273
...

```

## Inferencing
To sample the model:

```
./fetch.sh
python sample.py -n 64 -b 64 --checkpoint models/model.pt -o samples/ --image
```

