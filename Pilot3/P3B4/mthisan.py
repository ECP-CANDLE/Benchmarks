import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import math
import sys
import time
from sklearn.metrics import f1_score
import random
import logging
import yaml
import json

class MTHiSAN(nn.Module):
    '''
    multitask hierarchical self-attention network for classifying cancer pathology reports
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is masked, so the first row is ignored
      - num_classes: list[int]
        number of possible output classes for each task
      - max_words_per_line: int
        number of words per line
        used to split documents into smaller chunks
      - max_lines: int
        maximum number of lines per document
        additional lines beyond this limit are ignored
      - att_dim_per_head: int (default: 50)
        dimension size of output from each attention head
        total output dimension is att_dim_per_head * att_heads
      - att_heads: int (default: 8)
        number of attention heads for multihead attention
      - att_dropout: float (default: 0.1)
        dropout rate for attention softmaxes and intermediate embeddings
      - bag_of_embeddings: bool (default: False)
        adds a parallel bag of embeddings layer, concats to final doc embedding
    '''
    
    def __init__(self,
                 embedding_matrix,
                 num_classes,
                 max_words_per_line,
                 max_lines,
                 att_dim_per_head=50,
                 att_heads=8,
                 att_dropout=0.1,
                 bag_of_embeddings=False
                ):

        super(MTHiSAN,self).__init__()
        self.max_words_per_line = max_words_per_line
        self.max_lines = max_lines
        self.max_len = max_lines * max_words_per_line
        self.att_dim_per_head = att_dim_per_head
        self.att_heads = att_heads
        self.att_dim_total = att_heads * att_dim_per_head

        # normalize and initialize embeddings
        embedding_matrix -= embedding_matrix.mean()
        embedding_matrix /= (embedding_matrix.std()*2.5)
        embedding_matrix[0] = 0
        self.embedding = nn.Embedding.from_pretrained(
                         torch.tensor(embedding_matrix,dtype=torch.float),
                         freeze=False)
        self.word_embed_drop = nn.Dropout(p=att_dropout)

        # Q, K, V, and other layers for word-level self-attention
        self.word_q = nn.Linear(embedding_matrix.shape[1],self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_q.weight)
        self.word_q.bias.data.fill_(0.0)
        self.word_k = nn.Linear(embedding_matrix.shape[1],self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_k.weight)
        self.word_k.bias.data.fill_(0.0)
        self.word_v = nn.Linear(embedding_matrix.shape[1],self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.word_v.weight)
        self.word_v.bias.data.fill_(0.0)
        self.word_att_drop = nn.Dropout(p=att_dropout)        

        # target vector and other layers for word-level target attention
        self.word_target_drop = nn.Dropout(p=att_dropout)
        self.word_target = nn.Linear(1,self.att_dim_total,bias=False)
        torch.nn.init.uniform_(self.word_target.weight)
        self.line_embed_drop = nn.Dropout(p=att_dropout)

        # Q, K, V, and other layers for line-level self-attention
        self.line_q = nn.Linear(self.att_dim_total,self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_q.weight)
        self.line_q.bias.data.fill_(0.0)
        self.line_k = nn.Linear(self.att_dim_total,self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_k.weight)
        self.line_k.bias.data.fill_(0.0)
        self.line_v = nn.Linear(self.att_dim_total,self.att_dim_total)
        torch.nn.init.xavier_uniform_(self.line_v.weight)
        self.line_v.bias.data.fill_(0.0)

        # target vector and other layers for line-level target attention
        self.line_att_drop = nn.Dropout(p=att_dropout)
        self.line_target_drop =  nn.Dropout(p=att_dropout)
        self.line_target = nn.Linear(1,self.att_dim_total,bias=False)
        torch.nn.init.uniform_(self.line_target.weight)
        self.doc_embed_drop = nn.Dropout(p=att_dropout)

        # optional bag of embeddings layers
        self.boe = bag_of_embeddings
        if self.boe:
            self.boe_dense = nn.Linear(embedding_matrix.shape[1],embedding_matrix.shape[1])
            torch.nn.init.xavier_uniform_(self.boe_dense.weight)
            self.boe_dense.bias.data.fill_(0.0)
            self.boe_drop = nn.Dropout(p=0.5)

        # dense classification layers
        self.classify_layers = nn.ModuleList()
        for n in num_classes:
            in_size = self.att_dim_total
            if self.boe:
                in_size += embedding_matrix.shape[1]
            l = nn.Linear(in_size,n)
            torch.nn.init.xavier_uniform_(l.weight)
            l.bias.data.fill_(0.0)
            self.classify_layers.append(l)

    def _split_heads(self,x):
        '''
        splits final dim of tensor into multiple heads for multihead attention

        parameters:
          - x: torch.tensor (float) [batch_size x seq_len x dim]

        outputs:
          - torch.tensor (float) [batch_size x att_heads x seq_len x att_dim_per_head]
            reshaped tensor for multihead attention
        '''
        batch_size = x.size(0)
        x = x.view(batch_size,-1,self.att_heads,self.att_dim_per_head)
        return torch.transpose(x,1,2)
    
    def _attention(self,q,k,v,drop=None,mask_q=None,mask_k=None,mask_v=None):
        '''
        flexible attention operation for self and target attention
        
        parameters:
          - q: torch.tensor (float) [batch x heads x seq_len x dim1]
          - k: torch.tensor (float) [batch x heads x seq_len x dim1]
          - v: torch.tensor (float) [batch x heads x seq_len x dim2]
            NOTE: q and k must have the same dimension, but v can be different
          - drop: torch.nn.Dropout layer
          - mask_q: torch.tensor (bool) [batch x seq_len]
          - mask_k: torch.tensor (bool) [batch x seq_len]
          - mask_v: torch.tensor (bool) [batch x seq_len]
        '''

        # generate attention matrix
        scores = torch.matmul(q,torch.transpose(k,-1,-2))/math.sqrt(q.size(-1)) # batch x heads x seq_len x seq_len

        # this masks out empty entries in the attention matrix
        # and prevents the softmax function from assigning them any attention
        if mask_q is not None:
            mask_q = torch.unsqueeze(mask_q,1)
            mask_q = torch.unsqueeze(mask_q,-2)
            padding_mask = torch.logical_not(mask_q)
            scores -= 1.e7 * padding_mask.float()

        # normalize attention matrix
        weights = F.softmax(scores,-1)                                          # batch x heads x seq_len x seq_len

        # this removes empty rows in the normalized attention matrix
        # and prevents them from affecting the new output sequence
        if mask_k is not None:
            mask_k = torch.unsqueeze(mask_k,1)
            mask_k = torch.unsqueeze(mask_k,-1)
            weights = torch.mul(weights,mask_k.type(weights.dtype))

        # optional attention dropout
        if drop is not None:
            weights = drop(weights)

        # use attention on values to generate new output sequence
        result = torch.matmul(weights,v)                                        # batch x heads x seq_len x dim2

        # this applies padding to the entries in the output sequence
        # and ensures all padded entries are set to 0
        if mask_v is not None:
            mask_v = torch.unsqueeze(mask_v,1)
            mask_v = torch.unsqueeze(mask_v,-1)
            result = torch.mul(result,mask_v.type(result.dtype))

        return result

    def forward(self,docs,return_embeds=False):
        '''
        mthisan forward pass

        parameters:
          - docs: torch.tensor (int) [batch_size x words] 
            batch of documents to classify
            each document should be a 0-padded row of mapped word indices

        outputs:
          - list[torch.tensor (float) [batch_size x num_classes]]
            list of predicted logits for each task
        '''

        # bag of embeddings operations if enabled
        if self.boe:
            mask_words = (docs != 0)
            words_per_line = mask_words.sum(-1)
            max_words = words_per_line.max()
            mask_words = torch.unsqueeze(mask_words[:,:max_words],-1)
            docs_input_reduced = docs[:,:max_words]
            word_embeds = self.embedding(docs_input_reduced)
            word_embeds = torch.mul(word_embeds,mask_words.type(word_embeds.dtype))
            bag_embeds = torch.sum(word_embeds,1)
            bag_embeds = torch.mul(bag_embeds,
                         1/torch.unsqueeze(words_per_line,-1).type(bag_embeds.dtype))
            bag_embeds = torch.tanh(self.boe_dense(bag_embeds))
            bag_embeds = self.boe_drop(bag_embeds)

        # reshape into batch x lines x words
        docs = docs[:,:self.max_len]
        docs = docs.reshape(-1,self.max_lines,self.max_words_per_line)          # batch x max_lines x max_words

        # generate masks for word padding and empty lines
        # remove extra padding that exists across all documents in batch
        mask_words = (docs != 0)                                                # batch x max_lines x max_words
        words_per_line = mask_words.sum(-1)                                     # batch x max_lines
        max_words = words_per_line.max()                                        # hereon referred to as 'words'
        num_lines = (words_per_line != 0).sum(-1)                               # batch
        max_lines = num_lines.max()                                             # hereon referred to as 'lines'
        docs_input_reduced = docs[:,:max_lines,:max_words]                      # batch x lines x words
        mask_words = mask_words[:,:max_lines,:max_words]                        # batch x lines x words
        mask_lines = (words_per_line[:,:max_lines] != 0)                        # batch x lines

        # combine batch dim and lines dim for word level functions
        # also filter out empty lines for speedup and add them back in later
        batch_size = docs_input_reduced.size(0)
        docs_input_reduced = docs_input_reduced.reshape(
                             batch_size*max_lines,max_words)                    # batch*lines x words
        mask_words = mask_words.reshape(batch_size*max_lines,max_words)         # batch*lines x words
        mask_lines = mask_lines.reshape(batch_size*max_lines)                   # batch*lines
        docs_input_reduced = docs_input_reduced[mask_lines]                     # filtered x words
        mask_words = mask_words[mask_lines]                                     # filtered x words
        batch_size_reduced = docs_input_reduced.size(0)                         # hereon referred to as 'filtered'

        # word embeddings
        word_embeds = self.embedding(docs_input_reduced)                        # filtered x words x embed
        word_embeds = self.word_embed_drop(word_embeds)                         # filtered x words x embed
        
        # word self-attention
        word_q = F.elu(self._split_heads(self.word_q(word_embeds)))             # filtered x heads x words x dim
        word_k = F.elu(self._split_heads(self.word_k(word_embeds)))             # filtered x heads x words x dim
        word_v = F.elu(self._split_heads(self.word_v(word_embeds)))             # filtered x heads x words x dim
        word_att = self._attention(word_q,word_k,word_v,self.word_att_drop,
                       mask_words,mask_words,mask_words)                        # filtered x heads x words x dim

        # word target attention
        word_target = self.word_target(word_att.new_ones((1,1)))                
        word_target = word_target.view(
                      1,self.att_heads,1,self.att_dim_per_head)                 # 1 x heads x 1 x dim
        line_embeds = self._attention(word_target,word_att,word_att,
                      self.word_target_drop,mask_words)                         # filtered x heads x 1 x dim
        line_embeds = line_embeds.transpose(1,2).view(
                      batch_size_reduced,1,self.att_dim_total).squeeze(1)       # filtered x heads*dim
        line_embeds = self.line_embed_drop(line_embeds)                         # filtered x heads*dim

        # add in empty lines that were dropped earlier for line level functions
        line_embeds_full = line_embeds.new_zeros(
                           batch_size*max_lines,self.att_dim_total)             # batch*lines x heads*dim
        line_embeds_full[mask_lines] = line_embeds
        line_embeds = line_embeds_full
        line_embeds = line_embeds.reshape(
                      batch_size,max_lines,self.att_dim_total)                  # batch x lines x heads*dim
        mask_lines = mask_lines.reshape(batch_size,max_lines)                   # batch x lines

        # line self-attention
        line_q = F.elu(self._split_heads(self.line_q(line_embeds)))             # batch x heads x lines x dim
        line_k = F.elu(self._split_heads(self.line_k(line_embeds)))             # batch x heads x lines x dim
        line_v = F.elu(self._split_heads(self.line_v(line_embeds)))             # batch x heads x lines x dim
        line_att = self._attention(line_q,line_k,line_v,self.line_att_drop,
                   mask_lines,mask_lines,mask_lines)                            # batch x heads x lines x dim
        
        # line target attention
        line_target = self.line_target(line_att.new_ones((1,1)))
        line_target = line_target.view(
                      1,self.att_heads,1,self.att_dim_per_head)                 # 1 x heads x 1 x dim
        doc_embeds = self._attention(line_target,line_att,line_att,
                     self.line_target_drop,mask_lines)                          # batch x heads x 1 x dim
        doc_embeds = doc_embeds.transpose(1,2).view(
                     batch_size,1,self.att_dim_total).squeeze(1)                # batch x heads*dim
        doc_embeds = self.doc_embed_drop(doc_embeds)                            # batch x heads*dim

        # if bag of embeddings enabled, concatenate to hisan output
        if self.boe:
            doc_embeds = torch.cat([doc_embeds,bag_embeds],1)                   # batch x heads*dim+embed

        # generate logits for each task
        logits = []
        for l in self.classify_layers:
            logits.append(l(doc_embeds))                                        # batch x num_classes

        if return_embeds:
            return logits,doc_embeds
        return logits


