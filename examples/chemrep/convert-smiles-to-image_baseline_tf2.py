import argparse
import cairosvg
import io
import os
# import numpy as np
import pandas as pd

from PIL import Image, ImageOps
from rdkit import Chem
from rdkit.Chem import rdDepictor  # , RDConfig
from rdkit.Chem.Draw import rdMolDraw2D
import tensorflow.keras.backend as K

import convert_smiles as cs
import candle


def parse_args():
    parser = argparse.ArgumentParser(description='Encode SMILES to images')
    parser.add_argument('-f', '--filename', help="name of tab-delimited file for the list of SMILES")
    parser.add_argument('-o', '--out', help="output file name")
    parser.add_argument('--colname', default='SMILES', help="column name for SMILES")
    return parser.parse_args()


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """

    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            rgb = Image.merge('RGB', (r, g, b))
            inv = ImageOps.invert(rgb)
            r, g, b = inv.split()
            inv = Image.merge('RGBA', (r, g, b, a))
        elif img.mode == 'LA':
            l, a = img.split()
            l = ImageOps.invert(l)
            inv = Image.merge('LA', (l, a))
        else:
            inv = ImageOps.invert(img)
        return inv

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def smiles_to_image(mol, molSize=(512, 512), kekulize=True, mol_name='', mol_computed=False, invert=True):
    if not mol_computed:
        mol = Chem.MolFromSmiles(mol)
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except ValueError:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100, scale=1)))
    image.convert('RGB')
    return Invert()(image) if invert else image


def initialize_parameters(default_model='cs_default_model.txt'):
    csBmk = cs.BenchmarkConvertSmiles(cs.file_path, default_model, 'keras',
                                      prog='convert-smiles-to-image_baseline',
                                      desc='Convert SMILES to images')

    # Initialize parameters
    gParameters = candle.finalize_parameters(csBmk)

    return gParameters


def run(params):
    args = candle.ArgumentStruct(**params)

    try:
        out = args.out
    except AttributeError:
        out = args.filename + '.images'
    os.makedirs(out, exist_ok=True)
    print(f'Saving to {out}/\n')

    df = pd.read_csv(args.filename, sep='\t')
    drugs = df[args.colname]
    for index, smile in enumerate(drugs):
        print(f'{index}.png  <= ', 'SMILE:', smile)
        img = smiles_to_image(smile)
        img.save(f'{out}/{index}.png')


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
