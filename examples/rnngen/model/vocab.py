import argparse
import os
from tqdm import tqdm
from functools import partial
import multiprocessing
START_CHAR = '%'
END_CHAR = '^'


def get_vocab_from_file(fname):
    vocab = open(fname, 'r').readlines()
    vocab_ = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), vocab)))
    vocab_c2i = {k: v for v, k in enumerate(vocab_)}
    vocab_i2c = {v: k for v, k in enumerate(vocab_)}

    def i2c(i):
        return vocab_i2c[i]

    def c2i(c):
        return vocab_c2i[c]

    return vocab, c2i, i2c, vocab_c2i, vocab_i2c


#
# Generates a random permutations of smile strings.
#
def randomSmiles(smi, max_len=150, attempts=100):
    from rdkit import Chem
    import random

    def randomSmiles_(m1):
        m1.SetProp("_canonicalRankingNumbers", "True")
        idxs = list(range(0, m1.GetNumAtoms()))
        random.shuffle(idxs)
        for i, v in enumerate(idxs):
            m1.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(v))
        return Chem.MolToSmiles(m1)

    m1 = Chem.MolFromSmiles(smi)
    if m1 is None:
        return None
    if m1 is not None and attempts == 1:
        return [smi]

    s = set()
    for i in range(attempts):
        smiles = randomSmiles_(m1)
        s.add(smiles)
    s = list(filter(lambda x: len(x) < max_len, list(s)))

    if len(s) > 1:
        return s
    else:
        return [smi]


def main(args):
    if args.permute_smiles != 0:
        try:
            randomSmiles('CNOPc1ccccc1', 10)
        except Exception:
            print("Must set --permute_smiles to 0, cannot import RdKit. Smiles validity not being checked either.")

    # first step generate vocab.
    if args.start:
        count = 0
        vocab = set()
        vocab.update([START_CHAR, END_CHAR])
        print(vocab)
        with open(args.i, 'r') as f:
            for line in f:
                smi = line.strip()
                count += 1

                if len(smi) > args.maxlen - 2:
                    continue
                vocab.update(smi)

        vocab = list(vocab)
        with open(args.o + '/vocab.txt', 'w') as f:
            for v in vocab:
                f.write(v + '\n')

        print("Read ", count, "smiles.")
        print("Vocab length: ", len(vocab), "Max len: ", args.maxlen)

    count = 0

    _, c2i, _, _, _ = get_vocab_from_file(args.o + '/vocab.txt')

    # seconnd step is to make data:
    # count_ = count
    count = 0

    with open(args.i, 'r') as f:
        with open(args.o + '/out.txt', 'w') as o:
            with multiprocessing.Pool(args.p) as p:
                smiss = p.imap(partial(randomSmiles, max_len=args.maxlen, attempts=args.permute_smiles),
                               map(lambda x: x.strip(), f))
                for smis in tqdm(smiss):
                    if smis is None:
                        continue
                    for smi in smis:
                        if len(smi) > args.maxlen - 2:
                            continue
                        try:
                            i = list(map(lambda x: str(c2i(x)), smi))
                            if i is not None:
                                o.write(','.join(i) + '\n')
                                count += 1
                        except Exception:
                            print("key error did not print.", count)
                            continue
    print("Output", count, "smiles.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='input file with single smile per line')
    parser.add_argument('-o', type=str, required=True, help='output directory where preprocressed data and vocab will go')
    parser.add_argument('--start', action='store_true')
    parser.add_argument('--maxlen', type=int, required=True, help='max length for smile strings to be')
    parser.add_argument('--permute_smiles', type=int, help='generates permutations of smiles', default=0)
    parser.add_argument('-p', type=int, required=False, default=1)
    args = parser.parse_args()
    print(args)
    path = args.o
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed. Maybe it already exists? I will overwrite :)" % path)
    else:
        print("Successfully created the directory %s " % path)
    main(args)
