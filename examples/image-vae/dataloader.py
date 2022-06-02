import io

import cairosvg
import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torchvision import transforms

from invert import Invert


class MoleLoader(torch.utils.data.Dataset):
    def __init__(self, df, num=None):
        super(MoleLoader, self).__init__()

        size = df.shape[0]
        self.df = df.iloc[:int(size // 8), :]

        self.end_char = '?'

    def __len__(self):
        return self.df.shape[0]

    def make_image(self, mol, molSize=(256, 256), kekulize=True, mol_name=''):
        mol = Chem.MolFromSmiles(mol)
        mc = Chem.Mol(mol.ToBinary())
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except Exception:
                mc = Chem.Mol(mol.ToBinary())
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.SetFontSize(6)
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg, parent_width=100, parent_height=100,
                                                       scale=1)))
        image.convert('RGB')
        return Invert()(image)

    def get_vocab_len(self):
        return len(self.vocab)

    def generate_vocab(self):
        s = set(' ')
        for i, row in self.df.iterrows():
            s = s.union(row.iloc[0])
        print(s)
        self.vocab = list(s)

    def from_one_hot_array(self, vec):
        oh = np.where(vec == 1)
        if oh[0].shape == (0,):
            return None
        return int(oh[0][0])

    def decode_smiles_from_indexes(self, vec):
        return "".join(map(lambda x: self.charset[x], vec)).strip()

    def one_hot_array(self, i, n):
        return map(int, [ix == i for ix in range(n)])

    def one_hot_index(self, vec, charset):
        return map(charset.index, vec)

    def one_hot_encoded_fn(self, row):
        return np.array(map(lambda x: self.one_hot_array(x, self.vocab)),
                        self.one_hot_index(row, self.vocab))

    def apply_t(self, x):
        x = x + list((''.join([char * (self.embedding_width - len(x)) for char in [' ']])))
        smi = self.one_hot_encoded_fn(x)
        return smi

    def apply_one_hot(self, ch):
        return np.array(map(self.apply_t, ch))

    def __getitem__(self, item):
        smile = self.df.iloc[item, 0]
        smile_len = len(str(smile))
        image = self.make_image(smile)

        return 0, transforms.ToTensor()(image), smile_len
