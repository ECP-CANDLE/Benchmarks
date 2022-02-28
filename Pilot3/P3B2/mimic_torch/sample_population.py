import argparse
import numpy as np

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('input_prefix', help='location for data')
parser.add_argument('--number_of_samples', type=int, default=-1, help='number of samples')
parser.add_argument('--document_length', type=int, default=400, help='lenght of sampled documents')
parser.add_argument('--random', action='store_true', default=False, help='option to randomly sample documents')
parser.add_argument('--postfix', type=str, default='_samples', help='postfix for output file')

args = parser.parse_args()
print(args)

##########################################################################################
# 
##########################################################################################

number_tokens = 1
reverse_vocab = {}
with open(args.input_prefix + '_vocab.txt', 'r') as f:
    for row in f:
        number_tokens += 1
        reverse_vocab[row.split()[1]] = row.split()[0]

data_x = []
with open(args.input_prefix + '_train_x.txt', 'r') as  f:
    data_x = [x.strip() for x in f.readlines()]

if args.number_of_samples > 0:
    if args.random:
        sample_indices = np.random.choice(len(data_x), args.number_of_samples, replace=False)
        data_x = [data_x[i] for i in sample_indices]
    else:
        data_x = data_x[:args.number_of_samples]

with open(args.input_prefix + '_train_x' + args.postfix + '.txt', 'w') as f:
    with open(args.input_prefix + '_train_x' + args.postfix + '_str.txt', 'w') as g:
    
        g.write('document\n')
        
        for row in data_x:
            split_ids = row.split()
            split_tokens = [reverse_vocab[x] for x in split_ids]
            row_len = len(split_tokens)
            if row_len > args.document_length:
                start_index = np.random.choice(row_len-args.document_length+1)
                split_ids = split_ids[start_index:start_index+args.document_length]
                split_tokens = split_tokens[start_index:start_index+args.document_length]

            f.write('%s\n' % ' '.join(split_ids))
            g.write('%s\n' % ' '.join(split_tokens))

        