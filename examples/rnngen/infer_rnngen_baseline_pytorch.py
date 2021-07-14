import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn
import torch.utils.data

from model.model import CharRNN
from model.vocab import START_CHAR, END_CHAR
from model.vocab import get_vocab_from_file

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)


import candle

additional_definitions = [
    {'name': 'logdir',
        'type': str,
        'default': './',
        'help': 'place to store things.'},
    {'name': 'output',
        'type': str,
        'default': './',
        'help': 'place to store output smiles'},
    {'name': 'model',
        'type': str,
        'default': 'autosave.model.pt',
        'help': ''},
    {'name': 'temperature',
        'type': float,
        'default': 1.0,
        'help': 'temperature'},
    {'name': 'maxlen',
        'type': int,
        'default': 318,
        'help': ''},
    {'name': 'input',
        'type': str,
        'default': None,
        'help': 'Data from vocab folder'},
    {'name': 'nsamples',
        'type': int,
        'default': 1,
        'help': 'number samples to test'},
    {'name': 'vr',
        'type': candle.str2bool,
        'default': False,
        'help': 'validate, uses rdkit'},
    {'name': 'vb',
        'type': candle.str2bool,
        'default': False,
        'help': 'validate, uses openababel'},
    {'name': 'use_gpus',
        'type': candle.str2bool,
        'default': False,
        'help': ''},
]

required = [
    'batch_size',
    'logdir',
    'output',
    'input',
    'nsamples',
    'model',
]


class InferBk(candle.Benchmark):

    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters(default_model='infer_rnngen_default_model.txt'):

    # Build benchmark object
    sample = InferBk(file_path, default_model, 'pytorch',
                     prog='infer_rnngen_baseline',
                     desc='rnngen infer - Examples')

    print("Created sample benchmark")

    # Initialize parameters
    gParameters = candle.finalize_parameters(sample)
    print("Parameters initialized")

    return gParameters


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


def run(params):

    print("Note: This script is very picky. Please check device output to see where this is running. ")
    args = candle.ArgumentStruct(**params)

    data_url = args.data_url

    if args.model == 'ft_goodperforming_model.pt':
        file = 'pilot1/ft_goodperforming_model.pt'
    elif args.model == 'ft_poorperforming_model.pt':
        file = 'pilot1/ft_poorperforming_model.pt'
    else: # Corresponding to args.model == 'autosave.model.pt':
        file = 'mosesrun/autosave.model.pt'

    print('Recovering trained model')
    trained = candle.fetch_file(data_url + file, subdir='examples/rnngen')

    # Configure GPU
    if args.use_gpus and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Using device:', device)

    print("loading data.")
    vocab, c2i, i2c, _, _ = get_vocab_from_file(args.input + "/vocab.txt")

    model = CharRNN(len(vocab), len(vocab), max_len=args.maxlen).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Loading trained model.")
    pt = torch.load(trained, map_location=device)
    model.load_state_dict(pt['state_dict'])
    optimizer.load_state_dict(pt['optim_state_dict'])

    print("Applying to loaded data")
    total_sampled = 0
    total_valid = 0
    total_unqiue = 0
    smiles = set()
    start = time.time()

    batch_size = args.batch_size

    for epoch in range(int(args.nsamples / batch_size)):
        samples = sample(model, i2c, c2i, device, batch_size=batch_size, max_len=args.maxlen, temp=args.temperature)
        samples = list(map(lambda x: x[1:-1], samples))
        total_sampled += len(samples)
        if args.vr or args.vb:
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
    df.to_csv(args.output, index=False, header=True)

    print("output smiles to", args.output)
    print("Took ", end - start, "seconds")
    print("Sampled", total_sampled)
    print("Total unique", total_unqiue, float(total_unqiue) / float(total_sampled))
    if args.vr or args.vb:
        print("total valid", total_valid, float(total_valid) / float(total_sampled))


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
