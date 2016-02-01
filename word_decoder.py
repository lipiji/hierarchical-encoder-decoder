#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from lstm import *
from gru import *

class WordDecoderLayer(object):
    def __init__(self, cell, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "WordDecoderLayer_"
        layer_id = "_" + layer_id
        self.out_size, self.in_size = shape
        self.mask = mask
        self.X = X
        self.words = mask.shape[0]
        
        self.W_hy = init_weights((self.out_size, self.in_size), prefix + "W_hy" + layer_id)
        self.b_y = init_bias(self.in_size, prefix + "b_y" + layer_id)
        if cell == "gru":
            self.decoder = GRULayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask, is_train, batch_size, p)
            def _active(m, pre_h, x):
                x = T.reshape(x, (batch_size, self.in_size))
                pre_h = T.reshape(pre_h, (batch_size, self.out_size))

                h = self.decoder._active(x, pre_h)
                y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
                y = y * m[:, None]

                h = T.reshape(h, (1, batch_size * self.out_size))
                y = T.reshape(y, (1, batch_size * self.in_size))
                return h, y
            [h, y], updates = theano.scan(_active, #n_steps = self.words,
                                      sequences = [self.mask],
                                      outputs_info = [{'initial':self.X, 'taps':[-1]},
                                      T.alloc(floatX(0.), 1, batch_size * self.in_size)])
        elif cell == "lstm":
            self.decoder = LSTMLayer(rng, prefix + layer_id, (self.in_size, self.out_size), self.X, mask, is_train, batch_size, p)
            def _active(m, pre_h, pre_c, x):
                x = T.reshape(x, (batch_size, self.in_size))
                pre_h = T.reshape(pre_h, (batch_size, self.out_size))
                pre_c = T.reshape(pre_c, (batch_size, self.out_size))

                h, c = self.decoder._active(x, pre_h, pre_c)
            
                y = T.nnet.softmax(T.dot(h, self.W_hy) + self.b_y)
                y = y * m[:, None]

                h = T.reshape(h, (1, batch_size * self.out_size))
                c = T.reshape(c, (1, batch_size * self.out_size))
                y = T.reshape(y, (1, batch_size * self.in_size))
                return h, c, y
            [h, c, y], updates = theano.scan(_active, #n_steps = self.words,
                                             sequences = [self.mask],
                                             outputs_info = [{'initial':self.X, 'taps':[-1]},
                                                             {'initial':self.X, 'taps':[-1]},
                                                             T.alloc(floatX(0.), 1, batch_size * self.in_size)])
        
        y = T.reshape(y, (self.words, batch_size * self.in_size))
        self.activation = y
        self.params = self.decoder.params + [self.W_hy, self.b_y]

