#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T

from gru import *
from lstm import *
from word_encoder import *
from sent_encoder import *
from sent_decoder import *
from attention import *
from word_decoder import *
from updates import *

class RNN(object):
    def __init__(self, in_size, out_size, hidden_size,
                 cell = "gru", optimizer = "rmsprop", p = 0.5, num_sents = 1):

        self.X = T.matrix("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.cell = cell
        self.drop_rate = p
        self.num_sents = num_sents
        self.is_train = T.iscalar('is_train') # for dropout
        self.batch_size = T.iscalar('batch_size') # for mini-batch training
        self.mask = T.matrix("mask")
        self.optimizer = optimizer
        self.define_layers()
        self.define_train_test_funcs()
                
    def define_layers(self):
        self.layers = []
        self.params = []
        rng = np.random.RandomState(1234)
        # LM layers
        word_encoder_layer = WordEncoderLayer(rng, self.X, self.in_size, self.out_size, self.hidden_size,
                         self.cell, self.optimizer, self.drop_rate,
                         self.is_train, self.batch_size, self.mask)
        self.layers += word_encoder_layer.layers
        self.params += word_encoder_layer.params

        i = len(self.layers) - 1

        # encoder layer
        layer_input = word_encoder_layer.activation
        encoder_layer = SentEncoderLayer(self.cell, rng, str(i + 1), (word_encoder_layer.hidden_size, word_encoder_layer.hidden_size),
                                         layer_input, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(encoder_layer)
        self.params += encoder_layer.params
        
        # codes is a vector
        codes = encoder_layer.activation
        codes = T.reshape(codes, (1, encoder_layer.out_size))
        # sentence decoder
        sent_decoder_layer = SentDecoderLayer(self.cell, rng, str(i + 2), (encoder_layer.out_size, encoder_layer.in_size),
                                         codes, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(sent_decoder_layer)
        self.params += sent_decoder_layer.params

        # attention layer (syncrhonous update)
        sent_encs = encoder_layer.sent_encs
        sent_decs = sent_decoder_layer.activation 
        attention_layer = AttentionLayer(str(i + 3), (self.num_sents, sent_decoder_layer.out_size), sent_encs, sent_decs)
        
        # reshape to a row with num_sentences samples
        sents_codes = attention_layer.activation
        sents_codes = T.reshape(sents_codes, (1, self.batch_size * sent_decoder_layer.out_size))

        # word decoder
        word_decoder_layer = WordDecoderLayer(self.cell, rng, str(i + 4), (sent_decoder_layer.out_size, self.out_size),
                                         sents_codes, self.mask, self.is_train, self.batch_size, self.drop_rate)
        self.layers.append(word_decoder_layer)
        self.params += word_decoder_layer.params

        self.activation = word_decoder_layer.activation

    # https://github.com/fchollet/keras/pull/9/files
        self.epsilon = 1.0e-15
    def categorical_crossentropy(self, y_pred, y_true):
        y_pred = T.clip(y_pred, self.epsilon, 1.0 - self.epsilon)
        return T.nnet.categorical_crossentropy(y_pred, y_true).mean()
    
    def define_train_test_funcs(self):
        cost = self.categorical_crossentropy(self.activation, self.X)

        gparams = []
        for param in self.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        lr = T.scalar("lr")
        # eval(): string to function
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)

        #updates = sgd(self.params, gparams, lr)
        #updates = momentum(self.params, gparams, lr)
        #updates = rmsprop(self.params, gparams, lr)
        #updates = adagrad(self.params, gparams, lr)
        #updates = adadelta(self.params, gparams, lr)
        #updates = adam(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, self.mask, lr, self.batch_size],
                                               givens = {self.is_train : np.cast['int32'](1)},
                                               outputs = [cost, self.activation],
                                               updates = updates)
    
