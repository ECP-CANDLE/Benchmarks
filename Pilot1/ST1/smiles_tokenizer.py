import sys
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text

vocab_size = 40000  #
maxlen = 250  #

def read_csv_and_tokenize(fname):
    '''Takes a csv file as input. The csv file must have a header with
    values of "type", which is the docking score column, and "smiles",
    which is the smile string column'''

    data = pd.read_csv(fname, sep=',', index_col=None)
    print('done loading csv file\n', data.head(), '\n')

    y = data["type"].values.astype(float).reshape(-1, 1) * 1.0
    print('y shape {}\ny {}'.format(y.shape, y), '\n')

    tokenizer = text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data["smiles"])

    ## Tokenize and pad
    def prep_text(texts, tokenizer, max_sequence_length):
        # Turns text into into padded sequences.
        text_sequences = tokenizer.texts_to_sequences(texts)
        return sequence.pad_sequences(text_sequences, maxlen=maxlen)

    x = prep_text(data["smiles"], tokenizer, maxlen)
    print('x shape {}\nx {}'.format(x.shape, x), '\n')

    print('x type: {}, y type: {}'.format(type(x), type(y)))
    print('x shape: {}, y shape: {}'.format(x.shape, y.shape))
    return x, y


if __name__ == "__main__":
    fname=sys.argv[1]
    x, y = read_csv_and_tokenize(fname)
