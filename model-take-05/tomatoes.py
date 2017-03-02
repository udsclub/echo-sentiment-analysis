from __future__ import print_function

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Flatten
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

import six.moves.cPickle

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '../glove/' # http://nlp.stanford.edu/projects/glove/ pretrained vectors
TEXT_DATA_DIR = '../data/'
HEADER = True


import pandas as pd

df = pd.read_csv(TEXT_DATA_DIR + "reviews_rt_all.csv", delimiter='|')
X = df.text.as_matrix()
y = df.label.astype(str).as_matrix()

# TRAIN_LEN=75000
# TEST_LEN=25000

# X = []
# y = []

# X_train = []
# y_train = []
# 
# X_test = []
# y_test = []

# i=0
# with open(os.path.join(TEXT_DATA_DIR, "imdb_small.csv"), "r") as f:
#     if HEADER:
#         header = next(f)
#     for line in f:
#         temp_y, temp_x = line.rstrip("\n").split("|")
# 
#         X.append(temp_x)
#         y.append(temp_y)
# 
#         i+=1

# indices = np.arange(TRAIN_LEN)
# np.random.shuffle(indices)
# 
# for i in indices:
#     X_train.append(X[i])
#     y_train.append(y[i])
# 
# for i in range(TRAIN_LEN, TRAIN_LEN+TEST_LEN): #len(X)):
#     X_test.append(X[i])
#     y_test.append(y[i])

print("splitting data to train and test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 700
BATCH_SIZE=1024
EPOCHS=15


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS) # create dictionary of MAX_NB_WORDS, other words will not be used
tokenizer.fit_on_texts(X_train)
six.moves.cPickle.dump(tokenizer, open("tokenizer_rt.pkl", "wb"))

#X_train = np.concatenate((X_train[-10000:] , X_train[:-10000]))
#y_train = np.concatenate((y_train[-10000:] , y_train[:-10000]))

sequences = tokenizer.texts_to_sequences(X) # transform words to its indexes
sequences_train = tokenizer.texts_to_sequences(X_train) # transform words to its indexes
sequences_test = tokenizer.texts_to_sequences(X_test) # transform words to its indexes

word_index = tokenizer.word_index # dictionary of word:index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
                                                            # be careful because it takes only last MAX_SEQUENCE_LENGTH words
data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
                                                            
labels = to_categorical(np.asarray(y))
labels_train = to_categorical(np.asarray(y_train))
labels_test = to_categorical(np.asarray(y_test))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

if os.path.isfile("rt-model.h5"):
    print("Loading model...")
    model = load_model("rt-model.h5")
    print("done")
else:

    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Building model...')
    model = Sequential()
    #model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, dropout_W=0.2, dropout_U=0.2)))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])

    print('Train...')
    model.fit(data_train, labels_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data=(data_test, labels_test),)


    score, acc = model.evaluate(data_test, labels_test,
                                batch_size=BATCH_SIZE)

    print("score, acc")
    print("%.4f\t%.4f"%(score, acc))


    print("Saving model")
    model.save("rt-model.h5")
    print("done")

