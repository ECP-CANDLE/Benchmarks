import argparse
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn
import torch.utils.data

from model.model import CharRNN
from model.vocab import START_CHAR, END_CHAR
from train import get_vocab_from_file


def count_valid_samples(smiles, rdkit=True):
    if rdkit:
        from rdkit import Chem
        from rdkit import RDLogger
        lg = RDLogger.logger()

        lg.setLevel(RDLogger.CRITICAL)

        def toMol(smi):
            try:
                mol = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(mol)
            except Exception:
                return None
    else:
        import pybel

        def toMol(smi):
            try:
                m = pybel.readstring("smi", smi)
                return m.write("smi")
            except Exception:
                return None

    count = 0
    goods = []
    for smi in smiles:
        try:
            mol = toMol(smi)
            if mol is not None:
                goods.append(mol)
                count += 1
        except Exception:
            continue
    return count, goods


def sample(model, i2c, c2i, device, temp=1, batch_size=10, max_len=150):
    model.eval()
    with torch.no_grad():

        c_0 = torch.zeros((4, batch_size, 256)).to(device)
        h_0 = torch.zeros((4, batch_size, 256)).to(device)
        x = torch.tensor(c2i(START_CHAR)).unsqueeze(0).unsqueeze(0).repeat((max_len, batch_size)).to(device)

        eos_mask = torch.zeros(batch_size, dtype=torch.bool).to(device)
        end_pads = torch.tensor([max_len - 1]).repeat(batch_size).to(device)
        for i in range(1, max_len):
            x_emb = model.emb(x[i - 1, :]).unsqueeze(0)
            o, (h_0, c_0) = model.lstm(x_emb, (h_0, c_0))
            # o, h_0 = model.lstm(x_emb, h_0)
            y = model.linear(o.squeeze(0))
            y = F.softmax(y / temp, dim=-1)
            w = torch.multinomial(y, 1).squeeze()
            x[i, ~eos_mask] = w[~eos_mask]

            i_eos_mask = ~eos_mask & (w == c2i(END_CHAR))
            end_pads[i_eos_mask] = i + 1
            eos_mask = eos_mask | i_eos_mask

        new_x = []
        for i in range(x.size(1)):
            new_x.append(x[:end_pads[i], i].cpu())
        return ["".join(map(i2c, list(i_x.cpu().flatten().numpy()))) for i_x in new_x]


def main(args, device):
    # config = getconfig(args)
    print("loading data.")
    vocab, c2i, i2c, _, _ = get_vocab_from_file(args.i + "/vocab.txt")

    model = CharRNN(len(vocab), len(vocab), max_len=args.maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pt = torch.load(args.logdir + "/" + args.model, map_location=device)
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])

    total_sampled = 0
    total_valid = 0
    total_unqiue = 0
    smiles = set()
    start = time.time()

    batch_size = args.batch_size

    for epoch in range(int(args.n / batch_size)):
        samples = sample(model, i2c, c2i, device, batch_size=batch_size, max_len=args.maxlen, temp=args.t)
        samples = list(map(lambda x: x[1:-1], samples))
        total_sampled += len(samples)
        if args.vb or args.vr:
            valid_smiles, goods = count_valid_samples(samples, rdkit=args.vr)
            total_valid += valid_smiles
            smiles.update(goods)
        else:
            smiles.update(samples)
    smiles = list(smiles)
    total_unqiue += len(smiles)
    end = time.time()

    # with open(args.o, 'w') as f:
    #     for i in smiles:
    #         f.write(i)
    #         f.write('\n')

    df = pd.DataFrame()
    df['smiles'] = smiles
    df.to_csv(args.o, index=False, header=True)

    print("output smiles to", args.o)
    print("Took ", end - start, "seconds")
    print("Sampled", total_sampled)
    print("Total unique", total_unqiue, float(total_unqiue) / float(total_sampled))
    if args.vr or args.vb:
        print("total valid", total_valid, float(total_valid) / float(total_sampled))


if __name__ == '__main__':
    print("Note: This script is very picky. Please check device output to see where this is running. ")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Data from vocab folder', type=str, required=True)
    parser.add_argument('--logdir', help='place to store things.', type=str, required=True)
    parser.add_argument('-o', required=True, help='place to store output smiles', type=str)
    parser.add_argument('-n', help='number samples to test', type=int, required=True)
    parser.add_argument('-vr', help='validate, uses rdkit', action='store_true')
    parser.add_argument('-vb', help='validate, uses openababel', action='store_true')
    parser.add_argument('-t', help='temperature', default=1.0, required=False, type=float)
    parser.add_argument('--batch_size', default=128, required=False, type=int)
    parser.add_argument('--maxlen', default=318, required=False, type=int)
    parser.add_argument('--model', default='autosave.model.pt', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    main(args, device)
