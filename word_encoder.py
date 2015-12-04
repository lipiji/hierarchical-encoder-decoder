#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from gru import *
from lstm import *
from updates import *

class WordEncoderLayer(object):
    def __init__(self, rng, X, in_size, out_size, hidden_size,
                 cell, optimizer, p, is_train, batch_size, mask):
        self.X = X
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size_list = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.is_train = is_train
        self.batch_size = batch_size
        self.mask = mask
        self.rng = rng
        self.num_hds = len(hidden_size)

        self.define_layers()
    
    def define_layers(self):
        self.layers = []
        self.params = []
        # hidden layers
        for i in xrange(self.num_hds):
            if i == 0:
                layer_input = self.X
                shape = (self.in_size, self.hidden_size_list[0])
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.hidden_size_list[i - 1], self.hidden_size_list[i])

            if self.cell == "gru":
                hidden_layer = GRULayer(self.rng, str(i), shape, layer_input,
                                        self.mask, self.is_train, self.batch_size, self.drop_rate)
            elif self.cell == "lstm":
                hidden_layer = LSTMLayer(self.rng, str(i), shape, layer_input,
                                         self.mask, self.is_train, self.batch_size, self.drop_rate)
            
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        self.activation = hidden_layer.activation
        self.hidden_size = hidden_layer.out_size
