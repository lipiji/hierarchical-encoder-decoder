#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from gru import *
from lstm import *


class SentEncoderLayer(object):
    def __init__(self, cell, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "SentEncoder_"
        self.in_size, self.out_size = shape
        
        '''
        def code(j):
            i = mask[:, j].sum() - 1
            i = T.cast(i, 'int32')
            sent_x = X[i, j * self.in_size : (j + 1) * self.in_size]
            return sent_x
        sent_X, updates = theano.scan(lambda i: code(i), sequences=[T.arange(mask.shape[1])])
        '''
        sent_X = T.reshape(X[X.shape[0] - 1, :], (batch_size, self.in_size))
        mask = T.reshape(T.ones_like(sent_X)[:,0], (batch_size, 1))

        if cell == "gru":
            self.encoder = GRULayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, 1, p)
        elif cell == "lstm":
            self.encoder = LSTMLayer(rng, prefix + layer_id, shape, sent_X, mask, is_train, 1, p)
        
        self.activation = self.encoder.activation[self.encoder.activation.shape[0] - 1,:]
        self.sent_encs = sent_X
        self.params = self.encoder.params
