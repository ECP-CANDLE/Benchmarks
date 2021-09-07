

def getconfig(args):
    return args


def count_valid_samples(smiles):
    from rdkit import Chem
    count = 0
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi[1:-1])
        except Exception:
            continue
        if mol is not None:
            count += 1
    return count
