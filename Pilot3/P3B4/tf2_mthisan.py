import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from dense_attention import scaled_attention
import sys
import time
from sklearn.metrics import f1_score
import random


class History(object):
    def __init__(self):
        self.history = {}


class mthisan(object):

    class mthisan_model(Model):

        def __init__(self, embedding_matrix, num_classes, attention_size,
                     attention_heads):

            super(mthisan.mthisan_model, self).__init__()
            self.attention_size = attention_size
            self.attention_heads = attention_heads
            self.training = False

            self.embedding = layers.Embedding(embedding_matrix.shape[0],
                                              embedding_matrix.shape[1],
                                              embeddings_initializer=tf.keras.initializers.Constant(
                                              embedding_matrix.astype(np.float32)), trainable=False)

            self.word_drop = layers.Dropout(0.1)
            self.word_Q = layers.Dense(self.attention_size)
            self.word_K = layers.Dense(self.attention_size)
            self.word_V = layers.Dense(self.attention_size)
            self.word_target = tf.Variable(tf.random.uniform(shape=[1, self.attention_heads, 1,
                                                                    int(self.attention_size / self.attention_heads)]))
            self.word_self_att = scaled_attention(
                use_scale=1 / np.sqrt(attention_size), dropout=0.1)
            self.word_targ_att = scaled_attention(
                use_scale=1 / np.sqrt(attention_size), dropout=0.1)

            self.line_drop = layers.Dropout(0.1)
            self.line_Q = layers.Dense(self.attention_size)
            self.line_K = layers.Dense(self.attention_size)
            self.line_V = layers.Dense(self.attention_size)
            self.line_target = tf.Variable(tf.random.uniform(shape=[1, self.attention_heads, 1,
                                                                    int(self.attention_size / self.attention_heads)]))
            self.line_self_att = scaled_attention(
                use_scale=1 / np.sqrt(attention_size), dropout=0.1)
            self.line_targ_att = scaled_attention(
                use_scale=1 / np.sqrt(attention_size), dropout=0.1)

            self.doc_drop = layers.Dropout(0.1)

            self.classify_layers = []
            for c in num_classes:
                self.classify_layers.append(layers.Dense(c))

        def call(self, docs):

            # input shape: batch x lines x words
            batch_size = tf.shape(docs)[0]
            words_per_line = tf.math.count_nonzero(docs, 2, dtype=tf.int32)
            max_words = tf.reduce_max(words_per_line)
            lines_per_doc = tf.math.count_nonzero(
                words_per_line, 1, dtype=tf.int32)
            max_lines = tf.reduce_max(lines_per_doc)
            num_words = words_per_line[:, :max_lines]
            num_words = tf.reshape(num_words, (-1,))
            doc_input_reduced = docs[:, :max_lines, :max_words]

            # masks
            skip_lines = tf.not_equal(num_words, 0)
            count_lines = tf.reduce_sum(tf.cast(skip_lines, tf.int32))
            mask_words = tf.sequence_mask(num_words, max_words)[
                skip_lines]  # batch*max_lines x max_words
            # batch*max_lines x heads x max_words
            mask_words = tf.tile(tf.expand_dims(mask_words, 1), [
                                 1, self.attention_heads, 1])
            mask_lines = tf.sequence_mask(
                lines_per_doc, max_lines)  # batch x max_lines
            mask_lines = tf.tile(tf.expand_dims(mask_lines, 1), [
                                 1, self.attention_heads, 1])  # batch x heads x max_lines

            # word embeddings
            doc_input_reduced = tf.reshape(
                doc_input_reduced, (-1, max_words))[skip_lines]
            # batch*max_lines x max_words x embed_dim
            word_embeds = self.embedding(doc_input_reduced)
            word_embeds = self.word_drop(word_embeds, training=self.training)

            # word self attention
            # batch*max_lines x heads x max_words x depth
            word_q = self._split_heads(self.word_Q(word_embeds), count_lines)
            # batch*max_lines x heads x max_words x depth
            word_k = self._split_heads(self.word_K(word_embeds), count_lines)
            # batch*max_lines x heads x max_words x depth
            word_v = self._split_heads(self.word_V(word_embeds), count_lines)
            word_self_out = self.word_self_att([word_q, word_v, word_k],
                                               [mask_words, mask_words],
                                               training=self.training)

            # word target attention
            # batch*max_lines x heads x 1 x depth
            word_target = tf.tile(self.word_target, [count_lines, 1, 1, 1])
            word_targ_out = self.word_targ_att([word_target, word_self_out, word_self_out],
                                               [None, mask_words],
                                               training=self.training)
            # batch*max_lines x 1 x heads x depth
            word_targ_out = tf.transpose(word_targ_out, perm=[0, 2, 1, 3])
            line_embeds = tf.scatter_nd(tf.where(skip_lines),
                                        tf.reshape(
                                            word_targ_out, (count_lines, self.attention_size)),
                                        (batch_size * max_lines, self.attention_size))
            line_embeds = tf.reshape(
                line_embeds, (batch_size, max_lines, self.attention_size))
            line_embeds = self.line_drop(line_embeds, training=self.training)

            # line self attention
            # batch x heads x max_lines x depth
            line_q = self._split_heads(self.line_Q(line_embeds), batch_size)
            # batch x heads x max_lines x depth
            line_k = self._split_heads(self.line_K(line_embeds), batch_size)
            # batch x heads x max_lines x depth
            line_v = self._split_heads(self.line_V(line_embeds), batch_size)
            line_self_out = self.line_self_att([line_q, line_v, line_k],
                                               [mask_lines, mask_lines],
                                               training=self.training)

            # word target attention
            # batch x heads x 1 x depth
            line_target = tf.tile(self.line_target, [batch_size, 1, 1, 1])
            line_targ_out = self.line_targ_att([line_target, line_self_out, line_self_out],
                                               [None, mask_lines],
                                               training=self.training)
            # batch x 1 x heads x depth
            line_targ_out = tf.transpose(line_targ_out, perm=[0, 2, 1, 3])
            doc_embeds = tf.reshape(
                line_targ_out, (batch_size, self.attention_size))
            doc_embeds = self.doc_drop(doc_embeds, training=self.training)

            logits = []
            for lIndex in self.classify_layers:
                logits.append(lIndex(doc_embeds))
            return logits

        def _split_heads(self, x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.attention_heads,
                           int(self.attention_size / self.attention_heads)))
            return tf.transpose(x, perm=[0, 2, 1, 3])

    def __init__(self, embedding_matrix, num_classes, max_sents=201, max_words=15,
                 attention_heads=8, attention_size=400):

        self.ms = max_sents
        self.mw = max_words
        self.unk_tok = embedding_matrix.shape[0] - 1
        self.num_classes = num_classes
        self.num_tasks = len(num_classes)
        self.vocab_size = embedding_matrix.shape[0]
        self.model = self.mthisan_model(
            embedding_matrix, num_classes, attention_size, attention_heads)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(
            0.0001, 0.9, 0.99, 1e-08, False)
        # self.optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(self.optimizer)

    @tf.function
    def _train_step(self, text, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(text, training=True)
            loss = 0
            for i in range(self.num_tasks):
                loss += self.loss_object(labels[i],
                                         predictions[i]) / self.num_tasks
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return predictions, loss

    @tf.function
    def _score_step(self, text, labels):
        predictions = self.model(text, training=False)
        loss = 0
        for i in range(self.num_tasks):
            loss += self.loss_object(labels[i], predictions[i]) / self.num_tasks
        return predictions, loss

    @tf.function
    def _predict_step(self, text):
        predictions = self.model(text, training=False)
        return predictions

    def train(self, data, labels, batch_size=128, epochs=100, patience=5,
              validation_data=None, savebest=False, filepath=None):

        if savebest is True and filepath is None:
            raise Exception("Please enter a path to save the network")

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents'
              % (len(data), validation_size))

        history = History()

        # track best model for saving
        bestloss = np.inf
        pat_count = 0

        for ep in range(epochs):

            self.model.training = True

            # shuffle data
            labels.append(data)
            xy = list(zip(*labels))
            random.shuffle(xy)
            shuffled = list(zip(*xy))
            data = np.array(shuffled[-1]).astype(np.int32)
            labels = list(shuffled[:self.num_tasks])

            y_preds = [[] for c in self.num_classes]
            y_trues = [[] for c in self.num_classes]
            start_time = time.time()

            # train
            for start in range(0, len(data), batch_size):

                # get batch index
                if start + batch_size < len(data):
                    stop = start + batch_size
                else:
                    stop = len(data)

                # train step
                predictions, loss = self._train_step(data[start:stop],
                                                     np.array([lIndex[start:stop] for lIndex in labels]))

                # track correct predictions
                for i, (p, lIndex) in enumerate(zip(predictions, [lIndex[start:stop] for lIndex in labels])):
                    y_preds[i].extend(np.argmax(p, 1))
                    y_trues[i].extend(lIndex)
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"
                                 % (ep + 1, stop, len(data), loss))
                sys.stdout.flush()

            # checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time() - start_time))

            for i in range(self.num_tasks):
                micro = f1_score(y_trues[i], y_preds[i], average='micro')
                macro = f1_score(y_trues[i], y_preds[i], average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f"
                      % (ep + 1, i, micro, macro))

            scores, loss = self.score(validation_data[0], validation_data[1],
                                      batch_size=batch_size)

            for i in range(self.num_tasks):
                print("epoch %i task %i validation micro/macro: %.4f, %.4f"
                      % (ep + 1, i, scores[i][0], scores[i][1]))
            history.history.setdefault('val_loss', []).append(loss)

            # save if performance better than previous best
            if loss < bestloss:
                bestloss = loss
                pat_count = 0
                if savebest:
                    self.save(filepath)
            else:
                pat_count += 1
                if pat_count >= patience:
                    break

            # reset timer
            start_time = time.time()

        return history

    def predict(self, data, batch_size=128):

        self.model.training = False

        y_preds = [[] for c in self.num_classes]

        for start in range(0, len(data), batch_size):

            # get batch index
            if start + batch_size < len(data):
                stop = start + batch_size
            else:
                stop = len(data)

            predictions = self._predict_step(data[start:stop])
            for i, p in enumerate(predictions):
                y_preds[i].extend(np.argmax(p, 1))

            sys.stdout.write("processed %i of %i records        \r"
                             % (stop, len(data)))
            sys.stdout.flush()

        print()
        return y_preds

    def score(self, data, labels, batch_size=128):

        self.model.training = False

        y_preds = [[] for c in self.num_classes]
        losses = []

        for start in range(0, len(data), batch_size):

            # get batch index
            if start + batch_size < len(data):
                stop = start + batch_size
            else:
                stop = len(data)

            predictions, loss = self._score_step(data[start:stop],
                                                 [lIndex[start:stop] for lIndex in labels])
            for i, p in enumerate(predictions):
                y_preds[i].extend(np.argmax(p, 1))
            losses.append(loss)

            sys.stdout.write("processed %i of %i records        \r"
                             % (stop, len(data)))
            sys.stdout.flush()

        scores = []
        for i in range(self.num_tasks):
            micro = f1_score(labels[i], y_preds[i], average='micro')
            macro = f1_score(labels[i], y_preds[i], average='macro')
            scores.append([micro, macro])

        print()
        return scores, np.mean(losses)

    def save(self, savepath):

        self.model.save_weights(savepath)

    def load(self, savepath):

        self.model.load_weights(savepath)


if __name__ == "__main__":

    '''
    dummy test data
    '''

    # params
    batch_size = 32
    epochs = 5
    train_samples = 2000
    test_samples = 2000
    vocab_size = 750
    max_lines = 50
    max_words = 15
    num_classes = [2, 5, 10]
    embedding_size = 100
    attention_heads = 4
    attention_size = 128

    # create data
    vocab = np.random.rand(vocab_size, embedding_size)
    X = np.zeros((train_samples + test_samples, max_lines, max_words))
    for i, doc in enumerate(X):
        lIndex = np.random.randint(
            int(max_lines * max_words * 0.5), max_lines * max_words)
        row = np.zeros(max_lines * max_words)
        row[:lIndex] = np.random.randint(1, vocab_size, lIndex)
        X[i] = np.reshape(row, (max_lines, max_words))

    # test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = []
    for i in num_classes:
        y_trains.append(np.random.randint(0, i, train_samples))
    y_tests = []
    for i in num_classes:
        y_tests.append(np.random.randint(0, i, test_samples))

    # make save dir
    if not os.path.exists('savedmodels'):
        os.makedirs('savedmodels')

    # train model
    model = mthisan(vocab, num_classes, int(np.ceil(max_words / 15) + 1), 15,
                    attention_heads, attention_size)
    model.train(X_train, y_trains, batch_size, epochs,
                validation_data=(X_test, y_tests),
                savebest=True, filepath='savedmodels/model.ckpt')
#     model.load('savedmodels/model.ckpt')
