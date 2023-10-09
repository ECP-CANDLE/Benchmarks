import codecs
from SmilesPE.tokenizer import *
import pandas as pd
import time
from SmilesPE.learner import corpus_augment
#from SmilesPE.spe2vec import *
import sys
sys.path.append('SmilesPE/SmilesPE')
from spe2vec import *
from smiles_pair_encoders_functions import *

#smi = 'CC(=O)NCCC1=CNc2c1cc(OC)cc2CC(=O)NCCc1c[nH]c2ccc(OC)cc12'

if False:
    data_path = "/lus/grand/projects/datascience/avasan/Data_Docking/1M-flatten"
    rec = "3CLPro_7BQY_A_1_F"
    data = pd.read_csv(f'{data_path}/ml.{rec}.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.train')
    test_smi = data['smiles'][0:100]
    test_smi.to_csv('build_corpus.smi', index=False, header=False)

if False:
    infile = 'build_corpus.smi'
    outdir = 'data'
    corpus_augment(infile, outdir, cycles=1)

if False:
    spe_vocab = codecs.open('SmilesPE/SPE_ChEMBL.txt')
    spe = SPE_Tokenizer(spe_vocab)
    
    indir = 'data'
    corpus = Corpus(indir, tokenizer = spe, isdir=True, dropout=0.2)
    
    model = learn_spe2vec(corpus=corpus, n_jobs=64)
    model.save('spe_model.bin')

# some default tokens from huggingface
default_toks = ['[PAD]', 
                '[unused1]', '[unused2]', '[unused3]', '[unused4]','[unused5]', '[unused6]', '[unused7]', '[unused8]', '[unused9]', '[unused10]', 
                '[UNK]', '[CLS]', '[SEP]', '[MASK]']


# atom-level tokens used for trained the spe vocabulary
atom_toks = ['[c-]', '[SeH]', '[N]', '[C@@]', '[Te]', '[OH+]', 'n', '[AsH]', '[B]', 'b', 
             '[S@@]', 'o', ')', '[NH+]', '[SH]', 'O', 'I', '[C@]', '-', '[As+]', '[Cl+2]', 
             '[P+]', '[o+]', '[C]', '[C@H]', '[CH2]', '\\', 'P', '[O-]', '[NH-]', '[S@@+]', 
             '[te]', '[s+]', 's', '[B-]', 'B', 'F', '=', '[te+]', '[H]', '[C@@H]', '[Na]', 
             '[Si]', '[CH2-]', '[S@+]', 'C', '[se+]', '[cH-]', '6', 'N', '[IH2]', '[As]', 
             '[Si@]', '[BH3-]', '[Se]', 'Br', '[C+]', '[I+3]', '[b-]', '[P@+]', '[SH2]', '[I+2]', 
             '%11', '[Ag-3]', '[O]', '9', 'c', '[N-]', '[BH-]', '4', '[N@+]', '[SiH]', '[Cl+3]', '#', 
             '(', '[O+]', '[S-]', '[Br+2]', '[nH]', '[N+]', '[n-]', '3', '[Se+]', '[P@@]', '[Zn]', '2', 
             '[NH2+]', '%10', '[SiH2]', '[nH+]', '[Si@@]', '[P@@+]', '/', '1', '[c+]', '[S@]', '[S+]', 
             '[SH+]', '[B@@-]', '8', '[B@-]', '[C-]', '7', '[P@]', '[se]', 'S', '[n+]', '[PH]', '[I+]', '5', 'p', '[BH2-]', '[N@@+]', '[CH]', 'Cl']

fil = open('atom_toks.txt','w')
for  atom in atom_toks:
	fil.write(atom+"\n")
fil.close()


# spe tokens
with open('SmilesPE/SPE_ChEMBL.txt', "r") as ins:
    spe_toks = []
    for line in ins:
        spe_toks.append(line.split('\n')[0])

spe_tokens = []
for s in spe_toks:
    spe_tokens.append(''.join(s.split(' ')))
print('Number of SMILES:', len(spe_toks))

print(spe_tokens[:10])

spe_vocab = default_toks + atom_toks + spe_tokens
len(spe_vocab)

with open('vocab_spe.txt', 'w') as f:
    for voc in spe_vocab:
        f.write(f'{voc}\n')

tokenizer = SMILES_SPE_Tokenizer(vocab_file='vocab_spe.txt', spe_file= 'SmilesPE/SPE_ChEMBL.txt')

data_path = "/lus/grand/projects/datascience/avasan/Data_Docking/1M-flatten"
rec = "3CLPro_7BQY_A_1_F"
data_train = pd.read_csv(f'{data_path}/ml.{rec}.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.train')
test_smi = data['smiles']

lengths = []
maxlen = 38

from itertools import chain, repeat, islice

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)


for i,smi in enumerate(test_smi):
    token = tokenizer(smi)
    token['input_ids'] = list(pad(token['input_ids'], 38, 0))
    lengths.append(len(token['input_ids']))

    if i%10000 ==0:
        print(i)
        print(token['input_ids'])
        lengths[-1]

print(max(lengths))
#print([tokenizer(smi)['input_ids'] for smi in test_smi])



#start = time.time()
#data["smiles"] = [spe.tokenize(smi) for smi in data["smiles"]]
#end = time.time()
#
#print(f"{end - start} s")
#print(data["smiles"])
#print(spe.tokenize(smi))
