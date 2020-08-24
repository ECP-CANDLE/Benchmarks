import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score
import time
import sys

class History(object):
    def __init__(self):
        self.history = {}

class hisan(nn.Module):
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
                 attention_size=512,dropout=0.1,activation=None,lr=0.0001, optimizer= 'adam'):

        super(hisan, self).__init__()

        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.num_tasks = len(num_classes)
        self.embedding_size = embedding_matrix.shape[1]

        #embeddings
        self.embeddings = nn.Embedding(embedding_matrix.shape[0],self.embedding_size)
        self.embeddings.weight.requires_grad=False
        self.word_q = nn.Conv1d(self.embedding_size,self.attention_size,1)
        self.word_k = nn.Conv1d(self.embedding_size,self.attention_size,1)
        self.word_v = nn.Conv1d(self.embedding_size,self.attention_size,1)
        self.line_q = nn.Conv1d(self.attention_size,self.attention_size,1)
        self.line_k = nn.Conv1d(self.attention_size,self.attention_size,1)
        self.line_v = nn.Conv1d(self.attention_size,self.attention_size,1)

        #dropouts
        self.word_drop = nn.Dropout(p=dropout)
        self.line_drop = nn.Dropout(p=dropout)
        self.doc_drop = nn.Dropout(p=dropout)
        self.att_drop1 = nn.Dropout(p=dropout)
        self.att_drop2 = nn.Dropout(p=dropout)
        self.att_drop3 = nn.Dropout(p=dropout)
        self.att_drop4 = nn.Dropout(p=dropout)

        #classification
        self.softmaxes = torch.nn.ModuleList()
        for t in range(self.num_tasks):
            self.softmaxes.append(nn.Linear(attention_size,num_classes[t]))

        #initialize weights
        for name, param in self.named_parameters():
            try:
                nn.init.xavier_uniform_(param)
            except:
                nn.init.constant_(param,0)
        self.embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))

        #learnable target vectors
        self.word_target = nn.Parameter(torch.ones(1,1,attention_size))
        self.line_target = nn.Parameter(torch.ones(1,1,attention_size))

    def forward(self,documents):

        #generate document embeddings
        doc_embeds = []
        for doc in documents:
            doc_embeds.append(self._attention_step(doc).view(1,self.attention_size))
        doc_embeds = torch.cat(doc_embeds,0)

        classify = []
        for t in range(self.num_tasks):
            classify.append(self.softmaxes[t](doc_embeds))
        return classify

    def _attention_step(self,document):

        #get params and trim down
        num_words = np.count_nonzero(document,1)
        max_words = max(num_words)
        max_lines = np.count_nonzero(num_words)
        num_words = num_words[:max_lines]
        document = document[:max_lines,:max_words]

        #embedding lookup
        indices = Variable(torch.LongTensor(document))
        word_embeds = self.embeddings(indices)
        word_embeds = self.word_drop(word_embeds)

        #word position-wise feedforward
        Q = F.elu(self.word_q(torch.transpose(word_embeds,2,1)))
        K = F.elu(self.word_k(torch.transpose(word_embeds,2,1)))
        V = F.elu(self.word_v(torch.transpose(word_embeds,2,1)))
        Q = torch.transpose(Q,2,1)
        K = torch.transpose(K,2,1)
        V = torch.transpose(V,2,1)

        #word self attention
        outputs = torch.bmm(Q,torch.transpose(K,2,1))
        outputs = outputs/(self.attention_size**0.5)
        outputs = self.att_drop1(F.softmax(outputs,2))
        outputs = torch.bmm(outputs,V)

        #word target attention
        Q = self.word_target.repeat(max_lines,1,1)
        V = outputs

        outputs = torch.bmm(Q,torch.transpose(outputs,2,1))
        outputs = outputs/(self.attention_size**0.5)
        outputs = self.att_drop2(F.softmax(outputs,2))
        outputs = torch.bmm(outputs,V)
        line_embeds = torch.transpose(outputs,1,0)
        line_embeds = self.line_drop(line_embeds)

        #sent self attention
        Q = F.elu(self.line_q(torch.transpose(line_embeds,2,1)))
        K = F.elu(self.line_k(torch.transpose(line_embeds,2,1)))
        V = F.elu(self.line_v(torch.transpose(line_embeds,2,1)))
        Q = torch.transpose(Q,2,1)
        K = torch.transpose(K,2,1)
        V = torch.transpose(V,2,1)

        outputs = torch.bmm(Q,torch.transpose(K,2,1))
        outputs = outputs/(self.attention_size**0.5)
        outputs = self.att_drop3(F.softmax(outputs,2))
        outputs = torch.bmm(outputs,V)

        #sent target attention
        Q = self.line_target
        V = outputs

        outputs = torch.bmm(Q,torch.transpose(outputs,2,1))
        outputs = outputs/(self.attention_size**0.5)
        outputs = self.att_drop3(F.softmax(outputs,2))
        outputs = torch.bmm(outputs,V)
        doc_embed = self.doc_drop(outputs)

        return doc_embed.view(self.attention_size)

class hisan_gpu_trainer(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
                 attention_size=512,dropout_rate=0.1,activation=None,lr=0.0001, optimizer= 'adam',
                 ):

        model = hisan(embedding_matrix,num_classes,max_sents,max_words,
                     attention_size,dropout_rate)

        #model = torch.nn.DataParallel(model)
        #print('Model:', type(model))
        #print('Devices:', model.device_ids)
        # self.model = model.cuda()
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001)
        self.loss_func = nn.CrossEntropyLoss()
        self.num_tasks = len(num_classes)

    def train(self,data,labels,batch_size=100,epochs=50,validation_data=None):

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        history = History()

        for ep in range(epochs):

            y_preds = [[] for i in range(self.num_tasks)]
            y_trues = [[] for i in range(self.num_tasks)]
            start_time = time.time()

            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                self.optimizer.zero_grad()
                retvals = self.model(data[start:stop])

                #backprop step
                loss = 0
                for t in range(self.num_tasks):
                    target = Variable(torch.LongTensor(labels[t][start:stop]))
                    loss += self.loss_func(retvals[t],target)
                loss.backward()
                self.optimizer.step()

                #track correct predictions
                for t in range(self.num_tasks):
                    y_preds[t].extend(np.argmax(retvals[t].data.cpu().numpy(),1))
                    y_trues[t].extend(labels[t][start:stop])

                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (ep+1,stop,len(data),loss.data.cpu().numpy()))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))

            for t in range(self.num_tasks):
                micro = f1_score(y_trues[t],y_preds[t],average='micro')
                macro = f1_score(y_trues[t],y_preds[t],average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1,t+1,micro,macro))

            scores,val_loss = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            for t in range(self.num_tasks):
                print("epoch %i task %i validation micro/macro: %.4f, %.4f" % (ep+1,t+1,scores[t][0],scores[t][1]))
            history.history.setdefault('val_loss',[]).append(val_loss)

            #reset timer
            start_time = time.time()

        return history

    def score(self,data,labels,batch_size=16):

        loss = []
        y_preds = [[] for i in range(self.num_tasks)]
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            retvals = self.model(data[start:stop])

            #backprop step
            sum_loss = 0
            for t in range(self.num_tasks):
                target = Variable(torch.LongTensor(labels[t][start:stop]))
                sum_loss += self.loss_func(retvals[t],target)
            loss.append(float(sum_loss))

            for t in range(self.num_tasks):
                y_preds[t].extend(np.argmax(retvals[t].data.cpu().numpy(),1))

            sys.stdout.write("processed %i of %i records        \r" % (stop,len(data)))
            sys.stdout.flush()

        loss = np.mean(loss)

        print()
        scores = []
        for t in range(self.num_tasks):
            micro = f1_score(labels[t],y_preds[t],average='micro')
            macro = f1_score(labels[t],y_preds[t],average='macro')
            scores.append((micro,macro))

        return scores,loss

if __name__ == "__main__":

    import pickle
    from sklearn.model_selection import train_test_split

    #params
    batch_size = 4 * 16
    lr = 0.0001
    epochs = 3
    train_samples = 100
    test_samples = 10
    vocab_size = 100
    max_lines = 150
    max_words = 30
    num_classes = [5,2,10,3]
    embedding_size = 100

    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(0,vocab_size,(train_samples+test_samples,max_lines,max_words))

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = [np.random.randint(0,c,train_samples) for c in num_classes]
    y_tests = [np.random.randint(0,c,test_samples) for c in num_classes]

    #trainer model
    model = hisan_gpu_trainer(vocab,num_classes,max_lines,max_words)
    history = model.train(X_train,y_trains,batch_size=batch_size,epochs=epochs,
              validation_data=(X_test,y_tests))
    print(history.history)

