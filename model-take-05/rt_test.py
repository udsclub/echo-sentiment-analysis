
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Flatten
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Bidirectional

from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

import six.moves.cPickle

BASE_DIR = ''
TEXT_DATA_DIR = '../data/'
HEADER = True

X = []
y = []

i=0

with open(os.path.join(TEXT_DATA_DIR, "imdb_small.csv"), "r") as f:
    if HEADER:
        header = next(f)
    for line in f:
        line = line.strip()

        if line[0] not in ["0", "1"] or line[1] != "|":
            print("bad form of line", line)
            continue

        temp_y = line[0]
        temp_x = line[2:]

        X.append(temp_x)
        y.append(temp_y)
        
        i += 1

        
BATCH_SIZE=256
MAX_SEQUENCE_LENGTH=700

tokenizer = six.moves.cPickle.load(open("tokenizer_rt.pkl", "rb"))
sequences = tokenizer.texts_to_sequences(X) # transform words to its indexes

word_index = tokenizer.word_index # dictionary of word:index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
                                                            # be careful because it takes only last MAX_SEQUENCE_LENGTH words
labels = to_categorical(np.asarray(y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


model = load_model("rt-model.h5")
eval_results = model.evaluate(data, labels,
                            batch_size=BATCH_SIZE)

print('Test score:', eval_results[0])
print('Test accuracy:', eval_results[1:])

