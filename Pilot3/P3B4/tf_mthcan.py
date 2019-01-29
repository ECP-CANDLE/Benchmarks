import os
import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics import f1_score
import random

class History(object):
    def __init__(self):
        self.history = {}

class hcan(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
                 attention_size=512,dropout_rate=0.9,activation=tf.nn.elu,lr=0.0001, optimizer= 'adam'):

        tf.reset_default_graph()

        dropout_keep = dropout_rate

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.activation = activation
        self.num_tasks = len(num_classes)

        #doc input
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
        doc_embeds = tf.map_fn(self._attention_step,self.doc_input,dtype=tf.float32)

        #classification functions
        logits = []
        self.predictions = []
        for i in range(self.num_tasks):
            logit = tf.layers.dense(doc_embeds,num_classes[i],
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            logits.append(logit)
            self.predictions.append(tf.nn.softmax(logit))

        #loss, accuracy, and training functions
        self.labels = []
        self.loss = 0
        for i in range(self.num_tasks):
            label = tf.placeholder(tf.int32,shape=[None])
            self.labels.append(label)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=label))
            self.loss += loss/self.num_tasks
        # self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)
        if optimizer == 'adam':
             self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)
        elif optimizer == 'sgd':
             self.optimizer = tf.train.GradientDescentOptimizer( lr ).minimize( self.loss )
        elif optimizer == 'adadelta':
             self.optimizer = tf.train.AdadeltaOptimizer( learning_rate= lr ).minimize( self.loss )
        else:
             self.optimizer = tf.train.RMSPropOptimizer( lr ).minimize( self.loss )        

        #init op
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _attention_step(self,doc):

        words_per_line = tf.reduce_sum(tf.sign(doc),1)
        num_lines = tf.reduce_sum(tf.sign(words_per_line))
        max_words_ = tf.reduce_max(words_per_line)
        doc_input_reduced = doc[:num_lines,:max_words_]
        num_words = words_per_line[:num_lines]

        #word embeddings
        word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embedding_matrix,
                      dtype=tf.float32),doc_input_reduced)
        word_embeds = tf.nn.dropout(word_embeds,self.dropout)

        #masking
        mask_base = tf.cast(tf.sequence_mask(num_words,max_words_),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[1,1,max_words_])

        #word self attention
        Q = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.where(tf.equal(mask,0),tf.zeros_like(Q),Q)
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)
        V = tf.where(tf.equal(mask,0),tf.zeros_like(V),V)

        outputs = tf.matmul(Q,tf.transpose(K,[0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs),outputs)
        outputs = tf.matmul(outputs,V)
        outputs = tf.where(tf.equal(mask,0),tf.zeros_like(outputs),outputs)

        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        Q = tf.tile(Q,[num_lines,1,1])
        V = outputs

        outputs = tf.matmul(Q,tf.transpose(outputs,[0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V)
        sent_embeds = tf.transpose(outputs,[1,0,2])
        sent_embeds = tf.nn.dropout(sent_embeds,self.dropout)

        #sent self attention
        Q = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',
            activation=self.activation,kernel_initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.matmul(Q,tf.transpose(K,[0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V)

        #sent target attention       
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        V = outputs

        outputs = tf.matmul(Q,tf.transpose(outputs,[0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V)
        doc_embed = tf.nn.dropout(tf.squeeze(outputs,[0]),self.dropout)

        return tf.squeeze(doc_embed,[0])
    
    def train(self,data,labels,batch_size=100,epochs=50,validation_data=None):

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        history = History()

        for ep in range(epochs):

            #shuffle data
            labels.append(data)
            xy = list(zip(*labels))
            random.shuffle(xy)
            shuffled = list(zip(*xy))
            data = list(shuffled[-1])
            labels = list(shuffled[:self.num_tasks])

            y_preds = [[] for i in range(self.num_tasks)]
            y_trues = [[] for i in range(self.num_tasks)]
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.doc_input:data[start:stop],self.dropout:self.dropout_keep}
                for i in range(self.num_tasks):
                    feed_dict[self.labels[i]] = labels[i][start:stop]
                retvals = self.sess.run(self.predictions+[self.optimizer,self.loss],feed_dict=feed_dict)
                loss = retvals[-1]

                #track correct predictions
                for i in range(self.num_tasks):
                    y_preds[i].extend(np.argmax(retvals[i],1))
                    y_trues[i].extend(labels[i][start:stop])

                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop,len(data),loss))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            for i in range(self.num_tasks):
                micro = f1_score(y_trues[i],y_preds[i],average='micro')
                macro = f1_score(y_trues[i],y_preds[i],average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1,i+1,micro,macro))

            scores,val_loss = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            for i in range(self.num_tasks):
                print("epoch %i task %i validation micro/macro: %.4f, %.4f" % (ep+1,i+1,scores[i][0],scores[i][1]))
            history.history.setdefault('val_loss',[]).append(val_loss)

            #reset timer
            start_time = time.time()
            
        return history

    def predict(self,data,batch_size=100):
    
        y_preds = [[] for i in range(self.num_tasks)]
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input:data[start:stop],self.dropout:1.0}
            preds = self.sess.run(self.predictions,feed_dict=feed_dict)
            for i in range(self.num_tasks):
                y_preds[i].append(np.argmax(preds[i],1))

            sys.stdout.write("processed %i of %i records        \r" % (stop,len(data)))
            sys.stdout.flush()

        print()
        for i in range(self.num_tasks):
            y_preds[i] = np.concatenate(y_preds[i],0)
        return y_preds

    def score(self,data,labels,batch_size=16):
        
        loss = []
        y_preds = [[] for i in range(self.num_tasks)]
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input:data[start:stop],self.dropout:1.0}
            for i in range(self.num_tasks):
                feed_dict[self.labels[i]] = labels[i][start:stop]
            retvals = self.sess.run(self.predictions+[self.loss],feed_dict=feed_dict)
            loss.append(retvals[-1])
            
            for i in range(self.num_tasks):
                y_preds[i].append(np.argmax(retvals[i],1))

            sys.stdout.write("processed %i of %i records        \r" % (stop,len(data)))

            sys.stdout.flush()
        loss = np.mean(loss)

        print()
        for i in range(self.num_tasks):
            y_preds[i] = np.concatenate(y_preds[i],0)

        scores = []
        for i in range(self.num_tasks):
            micro = f1_score(labels[i],y_preds[i],average='micro')
            macro = f1_score(labels[i],y_preds[i],average='macro')
            scores.append((micro,macro))
        return scores,loss

    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    import pickle
    from sklearn.model_selection import train_test_split

    #params
    batch_size = 64
    lr = 0.0001
    epochs = 5
    train_samples = 500
    test_samples = 500
    vocab_size = 750
    max_lines = 150
    max_words = 30
    num_classes = [5,10,20,5,2]
    embedding_size = 100

    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(0,vocab_size,(train_samples+test_samples,max_lines,max_words))

    #optional masking
    min_lines = 30
    min_words = 5
    mask = []
    for i in range(train_samples+test_samples):
        doc_mask = np.ones((1,max_lines,max_words))
        num_lines = np.random.randint(min_lines,max_lines)
        for j in range(num_lines):
            num_words = np.random.randint(min_words,max_words)
            doc_mask[0,j,:num_words] = 0
        mask.append(doc_mask)

    mask = np.concatenate(mask,0)
    X[mask.astype(np.bool)] = 0

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = [np.random.randint(0,c,train_samples) for c in num_classes]
    y_tests = [np.random.randint(0,c,test_samples) for c in num_classes]

    #train model
    model = hcan(vocab,num_classes,max_lines,max_words,lr=lr)
    history = model.train(X_train,y_trains,batch_size=batch_size,epochs=epochs,
              validation_data=(X_test,y_tests))
    print(history.history)


