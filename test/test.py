
# pip instal keras tensorflow h5py

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

DATA_DIR = os.environ.get("DATA_DIR", '.')
INPUT_PATH= os.path.join(DATA_DIR, os.environ.get("INPUT_PATH", 'imdb-rt.csv'))

HEADER = True

X = []
y = []

with open(INPUT_PATH, "r") as f:
    if HEADER:
        header = next(f)

    for line in f:
        line = line.strip()

        if (line[0] not in ["0", "1"]) or (line[1] != "|"):
            continue

        y.append(line[0])
        X.append(line[2:])

        
BATCH_SIZE=256
MAX_SEQUENCE_LENGTH=220

tokenizer = six.moves.cPickle.load(open("tokenizer.pkl", "rb"))
# https://drive.google.com/open?id=0B8cauxN6J-tlcTVybWhXOFYzeFE

model = load_model("model.h5")
# https://drive.google.com/open?id=0B8cauxN6J-tlWmEybVRySFQwM2M

# $ md5sum model.h5 tokenizer.pkl
# a3c47bc1f5cf41955616942fd3a20032  model.h5
# 97b4ab4aac1e9600a9b92c97192b3ac0  tokenizer.pkl

sequences = tokenizer.texts_to_sequences(X) # transform words to its indexes

word_index = tokenizer.word_index # dictionary of word:index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(y))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


score, acc = model.evaluate(data, labels, batch_size=BATCH_SIZE)

print('Test score:', score)
print('Test accuracy:', acc)


