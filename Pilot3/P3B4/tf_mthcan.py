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

    def __init__(self, embedding_matrix, num_classes, max_sents, max_words,
                 attention_size=512, dropout_rate=0.9, activation=tf.nn.elu, lr=0.0001,
                 optimizer='adam', embed_train=True):

        tf.compat.v1.reset_default_graph()

        dropout_keep = dropout_rate

        self.dropout_keep = dropout_keep
        self.dropout = tf.compat.v1.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.activation = activation
        self.num_tasks = len(num_classes)
        self.embed_train = embed_train

        # doc input
        self.doc_input = tf.compat.v1.placeholder(tf.int32, shape=[None, max_sents, max_words])  # batch x sents x words
        batch_size = tf.shape(self.doc_input)[0]

        words_per_sent = tf.reduce_sum(tf.sign(self.doc_input), 2)  # batch X sents
        max_words_ = tf.reduce_max(words_per_sent)
        sents_per_doc = tf.reduce_sum(tf.sign(words_per_sent), 1)  # batch
        max_sents_ = tf.reduce_max(sents_per_doc)
        doc_input_reduced = self.doc_input[:, : max_sents_, : max_words_]  # clip

        doc_input_reshape = tf.reshape(doc_input_reduced, (-1, max_words_))  # batch*sents x words

        # word embeddings
        word_embeds = tf.gather(tf.compat.v1.get_variable('embeddings', initializer=self.embedding_matrix,
                                dtype=tf.float32, trainable=self.embed_train), doc_input_reshape)
        word_embeds = tf.nn.dropout(word_embeds, self.dropout)   # batch*sents x words x attention_size

        # word self attention
        Q = tf.layers.conv1d(word_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(word_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(word_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs)*-1000, outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
        outputs = tf.matmul(outputs, V)  # batch*sents x words x attention_size

        # word target attention
        Q = tf.compat.v1.get_variable('word_Q', (1, 1, self.attention_size),
                                      tf.float32, tf.orthogonal_initializer())
        Q = tf.tile(Q, [batch_size*max_sents_, 1, 1])
        V = outputs

        outputs = tf.matmul(Q, tf.transpose(outputs, [0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs)*-1000, outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
        outputs = tf.matmul(outputs, V)  # batch*sents x 1 x attention_size

        sent_embeds = tf.reshape(outputs, (-1, max_sents_, self.attention_size))
        sent_embeds = tf.nn.dropout(sent_embeds, self.dropout)  # batch x sents x attention_size

        # sent self attention
        Q = tf.layers.conv1d(sent_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(sent_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(sent_embeds, self.attention_size, 1, padding='same',
                             activation=self.activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs)*-1000, outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
        outputs = tf.matmul(outputs, V)  # batch x sents x attention_size

        # sent target attention
        Q = tf.compat.v1.get_variable('sent_Q', (1, 1, self.attention_size),
                                      tf.float32, tf.orthogonal_initializer())
        Q = tf.tile(Q, [batch_size, 1, 1])
        V = outputs

        outputs = tf.matmul(Q, tf.transpose(outputs, [0, 2, 1]))
        outputs = outputs/(K.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs)*-1000, outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
        outputs = tf.matmul(outputs, V)  # batch x 1 x attention_size
        doc_embeds = tf.nn.dropout(tf.squeeze(outputs, [1]), self.dropout)  # batch x attention_size

        # classification functions
        logits = []
        self.predictions = []
        for i in range(self.num_tasks):
            logit = tf.layers.dense(doc_embeds, num_classes[i],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
            logits.append(logit)
            self.predictions.append(tf.nn.softmax(logit))

        # loss, accuracy, and training functions
        self.labels = []
        self.loss = 0
        for i in range(self.num_tasks):
            label = tf.compat.v1.placeholder(tf.int32, shape=[None])
            self.labels.append(label)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=label))
            self.loss += loss/self.num_tasks

        if optimizer == 'adam':
            self.optimizer = tf.compat.v1.train.AdamOptimizer(lr, 0.9, 0.99)
        elif optimizer == 'sgd':
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
        elif optimizer == 'adadelta':
            self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=lr)
        else:
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)

        tf_version = tf.VERSION
        tf_version_split = tf_version.split('.')
        if(int(tf_version_split[0]) == 1 and int(tf_version_split[1]) > 13):
            self.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optimizer, loss_scale='dynamic')
        self.optimizer = self.optimizer.minimize(self.loss)

        # init op
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.compat.v1.train.Saver()
        self.sess = tf.compat.v1.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, data, labels, batch_size=100, epochs=50, validation_data=None):

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents'
              % (len(data), validation_size))

        history = History()

        for ep in range(epochs):

            # shuffle data
            labels.append(data)
            xy = list(zip(*labels))
            random.shuffle(xy)
            shuffled = list(zip(*xy))
            data = list(shuffled[-1])
            labels = list(shuffled[:self.num_tasks])

            y_preds = [[] for i in range(self.num_tasks)]
            y_trues = [[] for i in range(self.num_tasks)]
            start_time = time.time()

            # train
            for start in range(0, len(data), batch_size):

                # get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.doc_input: data[start: stop], self.dropout: self.dropout_keep}
                for i in range(self.num_tasks):
                    feed_dict[self.labels[i]] = labels[i][start: stop]
                retvals = self.sess.run(self.predictions+[self.optimizer, self.loss], feed_dict=feed_dict)
                loss = retvals[-1]

                # track correct predictions
                for i in range(self.num_tasks):
                    y_preds[i].extend(np.argmax(retvals[i], 1))
                    y_trues[i].extend(labels[i][start:stop])

                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"
                                 % (ep+1, stop, len(data), loss))
                sys.stdout.flush()

            # checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))

            for i in range(self.num_tasks):
                micro = f1_score(y_trues[i], y_preds[i], average='micro')
                macro = f1_score(y_trues[i], y_preds[i], average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1, i+1, micro, macro))

            scores, val_loss = self.score(validation_data[0], validation_data[1], batch_size=batch_size)
            for i in range(self.num_tasks):
                print("epoch %i task %i validation micro/macro: %.4f, %.4f" % (ep+1, i+1, scores[i][0], scores[i][1]))
            history.history.setdefault('val_loss', []).append(val_loss)

            # reset timer
            start_time = time.time()

        return history

    def predict(self, data, batch_size=100):

        y_preds = [[] for i in range(self.num_tasks)]
        for start in range(0, len(data), batch_size):

            # get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input: data[start: stop], self.dropout: 1.0}
            preds = self.sess.run(self.predictions, feed_dict=feed_dict)
            for i in range(self.num_tasks):
                y_preds[i].append(np.argmax(preds[i], 1))

            sys.stdout.write("processed %i of %i records        \r" % (stop, len(data)))
            sys.stdout.flush()

        print()
        for i in range(self.num_tasks):
            y_preds[i] = np.concatenate(y_preds[i], 0)
        return y_preds

    def score(self, data, labels, batch_size=16):

        loss = []
        y_preds = [[] for i in range(self.num_tasks)]
        for start in range(0, len(data), batch_size):

            # get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input: data[start: stop], self.dropout: 1.0}
            for i in range(self.num_tasks):
                feed_dict[self.labels[i]] = labels[i][start: stop]
            retvals = self.sess.run(self.predictions+[self.loss], feed_dict=feed_dict)
            loss.append(retvals[-1])

            for i in range(self.num_tasks):
                y_preds[i].append(np.argmax(retvals[i], 1))

            sys.stdout.write("processed %i of %i records        \r" % (stop, len(data)))

            sys.stdout.flush()
        loss = np.mean(loss)

        print()
        for i in range(self.num_tasks):
            y_preds[i] = np.concatenate(y_preds[i], 0)

        scores = []
        for i in range(self.num_tasks):
            micro = f1_score(labels[i], y_preds[i], average='micro')
            macro = f1_score(labels[i], y_preds[i], average='macro')
            scores.append((micro, macro))
        return scores, loss

    def save(self, filename):
        self.saver.save(self.sess, filename)

    def load(self, filename):
        self.saver.restore(self.sess, filename)
