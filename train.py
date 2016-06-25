# -*- coding: utf-8 -*-
#!/usr/bin/env

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
from six.moves import range
import os
import gc

from utils import progressBar, dataLoader, genVocabCreator


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#------- Global Variables ---------#
content_filepath   = 'new_data/out_TheSimpsons.tsv'
chars = '0123456789+/-*=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.,?(){}[]&#_ ' # Char-Level Vocabulary.
TRUNCATE_SIZE = 20000
#----------------------------------#

#--------- Load the context-response pairs -----------#
dataset = dataLoader(content_filepath) # access data as dataset.contexts, dataset.responses
print("[Data-Loader] : Loaded", len(dataset.contexts), "context-response pairs")
#-----------------------------------------------------#

#-------- Parameters for the model and dataset--------#
questions = dataset.contexts[:TRUNCATE_SIZE]
expected = dataset.responses[:TRUNCATE_SIZE]

RNN = recurrent.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 10
LAYERS = 3
X_MAXLEN = len(max(questions, key=len))
Y_MAXLEN = len(max(expected, key=len))

ctable = CharacterTable(chars, X_MAXLEN)
#-----------------------------------------------------#

#---------- Pre-process the data ---------------------#
print('[Pre-Process] : Performing data vectorization...')
X = np.zeros((len(questions), X_MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), Y_MAXLEN, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    X[i] = ctable.encode(sentence, maxlen=X_MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, maxlen=Y_MAXLEN)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
print('[Pre-Process] : Shuffling the data')
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
print('[Pre-Process] : Setting 10% data aside as validation-set')
split_at = len(X) - len(X) / 10
(X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
(y_train, y_val) = (y[:split_at], y[split_at:])

print("Question-Set Shape:",X_train.shape)
print("Response-Set Shape:",y_train.shape)
#-----------------------------------------------------#

#-------------- Define the Model ---------------------#
print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(X_MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(Y_MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))
#-----------------------------------------------------#

# Free some unused space.
gc.collect()

#--------------- Train the Model ---------------------#
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
        print('---')

    if iteration%10==0:
	gc.collect()
#-----------------------------------------------------#
