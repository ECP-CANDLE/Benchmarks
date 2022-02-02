import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import argparse

import numpy as np
import pandas as pd

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu


def parse_args():
    parser = argparse.ArgumentParser(description='Encode SMILES to latent vectors')
    parser.add_argument('-f', '--filename', help="name of tab-delimited file for the list of SMILES")
    parser.add_argument('-o', '--out', help="output folder name")
    parser.add_argument('--colname', default='SMILES', help="column name for SMILES")
    return parser.parse_args()


args = parse_args()
out = args.out or args.filename + '.encoded'

vae = VAEUtils(directory='models/zinc_properties')

df = pd.read_csv(args.filename, sep='\t')
drugs = df[args.colname]
latents = []
for smile in drugs:
    print('SMILE:', smile)
    if len(smile) > 120:
        s = ''
    else:
        x = vae.smiles_to_hot(smile, canonize_smiles=True)
        z = vae.encode(x)[0]
        # s = np.array2string(z, separator=',', suppress_small=True)
        s = str(z.tolist())
    latents.append(s)
    # print(z[:2], '...')
    # print()

df['Encoded'] = latents
df.to_csv(out, sep='\t')
