

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
GLOVE_DIR = BASE_DIR + '../../glove/' # http://nlp.stanford.edu/projects/glove/ pretrained vectors
TEXT_DATA_DIR = '../../data/'
HEADER = True


TRAIN_LEN=50000

X = []
y = []

i=0
with open(os.path.join(TEXT_DATA_DIR, "reviews_rt_all.csv"), "r") as f:
    if HEADER:
        header = next(f)
    for line in f:
        if i >= TRAIN_LEN:
            break
        temp_y, temp_x = line.rstrip("\n").split("|")

        X.append(temp_x)
        y.append(temp_y)

        
        i += 1


MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 40
BATCH_SIZE=32
EPOCHS=1

tokenizer = six.moves.cPickle.load(open("tokenizer.pkl", "rb"))
sequences = tokenizer.texts_to_sequences(X) # transform words to its indexes

word_index = tokenizer.word_index # dictionary of word:index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
                                                            # be careful because it takes only last MAX_SEQUENCE_LENGTH words
labels = to_categorical(np.asarray(y))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


model = load_model("tomatoes-model.h5")
score, acc = model.evaluate(data, labels,
                            batch_size=BATCH_SIZE)

print('Test score:', score)
print('Test accuracy:', acc)


