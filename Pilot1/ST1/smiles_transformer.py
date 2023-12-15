from __future__ import print_function

import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence, text

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "embed_dim", "type": int, "help": "Embedding dimension for each token"},
    {
        "name": "ff_dim",
        "type": int,
        "help": "Hidden layer size in feed forward network inside transformer",
    },
    {"name": "maxlen", "type": int, "help": "Maximum sequence length"},
    {"name": "num_heads", "type": int, "help": "Number of attention heads"},
    {"name": "out_layer", "type": int, "help": "Size of output layer"},
    {"name": "transformer_depth", "type": int, "help": "Number of transformer layers"},
    {"name": "vocab_size", "type": int, "help": "Vocabulary size"},
]

required = []


class BenchmarkST(candle.Benchmark):
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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# define r2 for reporting


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def prep_text(texts, tokenizer, max_sequence_length):
    # Turns text into into padded sequences.
    text_sequences = tokenizer.texts_to_sequences(texts)
    return sequence.pad_sequences(text_sequences, maxlen=max_sequence_length)


def load_data(params):

    data_path_train = candle.fetch_file(
        params["data_url"] + params["train_data"], "Pilot1"
    )
    data_path_val = candle.fetch_file(params["data_url"] + params["val_data"], "Pilot1")

    vocab_size = params["vocab_size"]
    maxlen = params["maxlen"]

    data_train = pd.read_csv(data_path_train)
    data_vali = pd.read_csv(data_path_val)

    data_train.head()

    # Dataset has type and smiles as the two fields

    y_train = data_train["type"].values.reshape(-1, 1) * 1.0
    y_val = data_vali["type"].values.reshape(-1, 1) * 1.0

    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data_train["smiles"])

    x_train = prep_text(data_train["smiles"], tokenizer, maxlen)
    x_val = prep_text(data_vali["smiles"], tokenizer, maxlen)

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train, x_val, y_val


def transformer_model(params):

    embed_dim = params["embed_dim"]  # 128
    ff_dim = params["ff_dim"]  # 128
    maxlen = params["maxlen"]  # 250
    num_heads = params["num_heads"]  # 16
    vocab_size = params["vocab_size"]  # 40000
    transformer_depth = params["transformer_depth"]  # 4

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    for i in range(transformer_depth):
        x = transformer_block(x)

    # = layers.GlobalAveragePooling1D()(x)  --- the original model used this but the accuracy was much lower

    dropout = params["dropout"]  # 0.1
    dense_layers = params["dense"]  # [1024, 256, 64, 16]
    activation = params["activation"]  # 'relu'
    out_layer = params["out_layer"]  # 2 for class, 1 for regress
    out_act = params["out_activation"]  # 'softmax' for class, 'relu' for regress

    x = layers.Reshape((1, 32000), input_shape=(250, 128,))(
        x
    )  # reshaping increases parameters but improves accuracy a lot
    x = layers.Dropout(0.1)(x)
    for dense in dense_layers:
        x = layers.Dense(dense, activation=activation)(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(out_layer, activation=out_act)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.summary()

    return model
