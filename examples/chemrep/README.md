## Converting small molecule SMILES to autoencoded vectors or images

Utility scripts for converting SMILES to other representations that may be useful as input drug features in deep learning models.

### Example 1

Adding a new column of 192-dimensional learned vector representations for the drug SMILES used in the Pilot1 project. The trained autoencoder model is from [ChemVAE](README.chemvae.md) trained on the ZINC database.

```
python convert-smiles-to-latent.py -f pilot1-drugs
```

In this example, `pilot1-drugs` is a tab-delimited file with a `SMILES` column. The `--colname` argument can be used to select the column in the input file for SMILES information.

### Example 2

Creating images for SMILES. 

```
python convert-smiles-to-image.py -f pilot1-drugs
```
This will create an output folder named `pilot1-drugs.images` with the converted images for the drugs in the original tab-delimited file.

### Installation

Installation with a conda environment is recommended.
```
conda env create -f environment.yml
source activate chemrep
python setup.py install
```
