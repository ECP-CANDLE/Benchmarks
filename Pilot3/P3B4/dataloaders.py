import torch
from torch.utils.data import Dataset, DataLoader, Subset
import sklearn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import entropy
import random
import math
import ast

class addNoise(object):
    '''
    optional transform object for PathReports dataset
      - adds random amount of padding at front of document using unk_tok to
        reduce hisan overfitting 
      - randomly replaces words with randomly selected other words to reduce
        overfitting

    parameters:
      - unk_token: int
        integer mapping for unknown tokens
      - max_pad_len: int
        maximum amount of padding at front of document
      - vocab_size: int
        size of vocabulary matrix or
        maximum integer value to use when randomly replacing word tokens
      - switch_rate: float (default: 0.1)
        percentage of words to randomly replace with random tokens        
    '''

    def __init__(self,unk_token,max_pad_len,vocab_size,switch_rate=0.1):
        self.unk_token = unk_token
        self.max_pad_len = max_pad_len
        self.vocab_size = vocab_size
        self.switch_rate = switch_rate

    def __call__(self, doc):
        pad_amt = np.random.randint(0,self.max_pad_len)
        doc = [int(self.unk_token) for i in range(pad_amt)] + list(doc)
        l = len(doc)
        r_idx = np.random.choice(np.arange(l),int(l*self.switch_rate),replace=False)
        r_voc = np.random.randint(1,self.vocab_size,len(r_idx))
        doc = np.array(doc)
        doc[r_idx] = r_voc
        return doc

class LabelSmoothing(object):
    def __init__(self,confusion_matrix):
        pass

class PathReports(Dataset):
    '''
    torch dataset for cancer path reports from generate_data.py

    parameters:
      - np_X: np.ndarray
        numpy array tokenized path report data, generated from generate_data.py
      - df_Y: pd.DataFrame
        dataframe raw truth values
      - tasks: list[string]
        list of tasks to generate labels for
      - label_encoders: dict[str,sklearn.preprocessing.LabelEncoder]
        dict (task:label encoders) to convert raw labels into integers
      - max_len: int (default: 3000)
        maximum length for document, should match value in data_args.json
        longer documents will be cut, shorter documents will be 0-padded
      - transform: object (default: None)
        optional transform to apply to document tensors
      - label_smoothing: object (default: None)
        not implemented yet

    outputs per batch:
      - dict[str:torch.tensor]
        sample dictionary with following keys/vals:
          - 'X': torch.tensor (int) [max_len]
            document converted to integer word-mappings, 0-padded to max_len
          - 'y_%s % task': torch.tensor (int) [] or
                           torch.tensor (int) [num_classes]
            integer label for a given task if label encoders are used
            one hot vectors for a given task if label binarizers are used
    '''

    def __init__(self, 
                 np_X,
                 df_Y, 
                 tasks, 
                 label_encoders,
                 max_len=3000, 
                 transform=None,
                 mutual_info_filter=False,
                 mutual_info_threshold=0.005,
                 label_smoothing=None,
                 multilabel=False,
                 soft_Y= None,
                ):
    
        self.X = np_X
        self.ys = {}
        self.ys_onehot = {}
        self.label_encoders = label_encoders
        self.num_classes = {}
        self.tasks = tasks
        self.transform = transform
        self.max_len = max_len
        self.multilabel = multilabel
        #if soft_Y is not None:
        #    self.softlabel = True
        #self.soft_Y = soft_Y

        for t, task in enumerate( tasks ):
            y = df_Y[task]
            #le = {v:int(k) for k,v in self.label_encoders[task].items()}
            #y = y.apply(lambda v: le[v]).to_numpy()
            le = self.label_encoders[ task ]
            y = le.transform( y )
            #ignore abstention class if it exists
            #if 'abs_%s' % task in le:
            #    del le['abs_%s' % task]
            self.num_classes[task] = len(le.classes_)
            self.ys[task] = y

            y_onehot = np.zeros((len(y),len(le.classes_)))
            y_onehot[np.arange(len(y)),y] = 1

            if soft_Y is not None:
                y_onehot = soft_Y[ t ]
            self.ys_onehot[task] = y_onehot

            
            

        # apply token filter if enabled
        if type(mutual_info_filter) == np.ndarray:
            self.mi_values = self._mutual_information_filter(
                             mutual_info_threshold,mutual_info_filter)
        elif mutual_info_filter == True:
            self.mi_values = self._mutual_information_filter(
                             mutual_info_threshold)
        else:
            self.mi_values = None

    def _mutual_information_filter(self,threshold,mi_values=None):

        # convert docs into sparse matrices
        if mi_values is None:
            print('converting docs to sparse matrices')
            doc_lens = [0]
            total_doc_len = 0
            for doc in self.X:
                total_doc_len += len(doc)
                doc_lens.append(total_doc_len)
            sparse_docs = csr_matrix((np.ones(total_doc_len),
                                      [w for doc in self.X for w in doc],
                                      doc_lens),dtype=int)
            vocab_size = sparse_docs.shape[1]

            # calculate mutual information by token for each task
            mi_values = np.zeros(vocab_size)
            for task in self.tasks:
                print('calculating mutual information for %s' % task)
                mi_values_ = mutual_info_classif(sparse_docs,
                                                 self.ys[task],
                                                 discrete_features=True)
                ys = np.sum(self.ys_onehot[task],0)
                #mi_values_ /= entropy(ys)
                mi_values = np.maximum(mi_values,mi_values_)

        # filter tokens by mutual information
        mi_filter = mi_values >= threshold
        mi_filter = np.nonzero(mi_filter)[0]
        print('retaining %.4f of total tokens = %i tokens' % \
              (len(mi_filter)/len(mi_values),len(mi_filter)))
        for i,doc in enumerate(self.X):
            self.X[i] = list(np.array(doc)[np.isin(doc,mi_filter)])
        return mi_values

    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        doc = self.X[idx]
        if self.transform:
            doc = self.transform(doc)
        array = np.zeros(self.max_len)
        doc = doc[:self.max_len]
        l = len(doc)
        array[:l] = doc
        sample = {'X':torch.tensor(array,dtype=torch.long)}
        for t,task in enumerate(self.tasks):
            if self.multilabel:
                y = self.ys_onehot[task][idx]
                sample['y_%s' % task] = torch.tensor(y,dtype=torch.float)
                #elif self.softlabel:
                #    y = self.soft_Y[ t ][ idx ]
                #    sample[ 'y_%s' % task ] = torch.tensor( y, dtype= torch.float )
            else:
                y = self.ys[task][idx]
                sample['y_%s' % task] = torch.tensor(y,dtype=torch.long)
        return sample

class grouped_cases(Dataset):
    def __init__(self,
                 doc_embeds,
                 df_Y,
                 tasks,
                 metadata,
                 label_encoders,
                 exclude_single=False,
                 shuffle_case_order=False,
                 split_by_tumorid=False,
                ):

        self.embed_size = len(doc_embeds[0])
        self.tasks = tasks
        self.shuffle_case_order = shuffle_case_order
        self.label_encoders = {}
        for task in tasks:
            le = {v:int(k) for k,v in label_encoders[task].items()}
            self.label_encoders[task] = le
        self.grouped_X = []
        self.grouped_ys = {task:[] for task in tasks}
        self.new_idx = []
        if split_by_tumorid:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str) + metadata['tumorId'].astype(str)
        else:
            metadata['uid'] = metadata['registryId'] + metadata['patientId'].astype(str)
        groups = metadata.reset_index().groupby('uid')
        self.max_seq_len = groups.agg('count').max().tolist()[0]
        try:
            df_Y_ = df_Y.reset_index()
        except:
            print('Warning: skipping df_y.reset_index()')

        for i,(name,group) in enumerate(groups):

            if exclude_single and len(group.index) == 1:
                continue

            g = group.sort_values(by='recordDocumentId')
            group = []
            indices = []
            labels = {task:[] for task in tasks}
            for idx in g.index:
                group.append(doc_embeds[idx])
                indices.append(idx)
                for task in tasks:
                    label = df_Y_[task][idx]
                    labels[task].append(self.label_encoders[task][label])
            self.grouped_X.append(np.vstack(group))
            self.new_idx.append(indices)
            for task in tasks:
                self.grouped_ys[task].append(labels[task])

    def __len__(self):
        return len(self.grouped_X)

    def __getitem__(self,idx):

        seq = self.grouped_X[idx]
        ys = []
        for t,task in enumerate(self.tasks):
            y = self.grouped_ys[task][idx]
            ys.append(y)
        if self.shuffle_case_order:
            ys = np.array(ys).T
            shuffled = list(zip(seq,ys))
            random.shuffle(shuffled)
            seq,ys = zip(*shuffled)
            seq = np.array(seq)
            ys = np.array(ys).T

        array = np.zeros((self.max_seq_len,self.embed_size))
        l = len(seq)
        array[:l,:] = seq
        sample = {'X':torch.tensor(array,dtype=torch.float)}
        sample['len'] = torch.tensor(l,dtype=torch.long)

        for t,task in enumerate(self.tasks):
            y = ys[t]
            array = np.zeros(self.max_seq_len)
            array[:l] = y
            sample['y_%s' % task] = torch.tensor(array,dtype=torch.long)

        indices = self.new_idx[idx]
        array = np.zeros(self.max_seq_len)
        array[:l] = indices
        sample['new_idx'] = torch.tensor(array,dtype=torch.long)
        return sample

