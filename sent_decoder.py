#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from lstm import *
from gru import *

class SentDecoderLayer(object):
    def __init__(self, cell, rng, layer_id, shape, X, mask, sent_enc, is_train = 1, batch_size = 1, p = 0.5, num_sents = 1):
        prefix = "SentDecoderLayer_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size, self.sent_enc_size = shape
        self.X = X
        self.summs = batch_size
        
        self.W_hy = init_weights((self.in_size, self.out_size), prefix + "W_hy" + layer_id)
        self.b_y = init_bias(self.out_size, prefix + "b_y" + layer_id)

        self.W_a1 = init_weights((self.out_size, self.out_size), prefix + "W_a1" + layer_id)
        self.W_a2 = init_weights((self.sent_enc_size, self.out_size), prefix + "W_a2" + layer_id)
        self.W_a3 = init_weights((self.out_size, self.out_size), prefix + "W_a3" + layer_id)
        self.W_a4 = init_weights((self.out_size, self.out_size), prefix + "W_a4" + layer_id)
        self.U_a = init_weights((self.out_size, num_sents), prefix + "U_a" + layer_id)

        if cell == "gru":
            self.decoder = GRULayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask, is_train, 1, p)
            def _active(pre_h, x):
                h = self.decoder._active(x, pre_h)
                y = T.tanh(T.dot(h, self.W_hy) + self.b_y)
                return h, y
            [h, y], updates = theano.scan(_active, n_steps = self.summs, sequences = [],
                                      outputs_info = [{'initial':self.X, 'taps':[-1]},
                                                      T.alloc(floatX(0.), 1, self.out_size)])
        elif cell == "lstm":
            self.decoder = LSTMLayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask, is_train, 1, p)
            def _active(pre_h, pre_c, x):
                h, c = self.decoder._active(x, pre_h, pre_c)
                y = T.tanh(T.dot(h, self.W_hy) + self.b_y)
                return h, c, y
            [h, c, y], updates = theano.scan(_active, n_steps = self.summs, sequences = [],
                                             outputs_info = [{'initial':self.X, 'taps':[-1]},
                                                             {'initial':self.X, 'taps':[-1]},
                                                             T.alloc(floatX(0.), 1, self.out_size)])
       
        y = T.reshape(y, (self.summs, self.out_size))
        strength = T.dot(T.nnet.sigmoid(T.dot(y, self.W_a1) + T.dot(sent_enc, self.W_a2)), self.U_a)
        attention = T.dot(T.nnet.softmax(strength), sent_enc)
        self.activation = T.dot(y, self.W_a3) + T.dot(attention, self.W_a4)

        self.params = self.decoder.params + [self.W_hy, self.b_y] + [self.W_a1, self.W_a2, self.W_a3, self.W_a4, self.U_a]

