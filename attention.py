#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class AttentionLayer(object):
    def __init__(self, layer_id, shape, sent_encs, sent_decs):
        prefix = "AttentionLayer_"
        layer_id = "_" + layer_id
        self.num_sents, self.out_size = shape

        # TODO fix the attention layer using standard attentnion modeling method
        self.W_a1 = init_weights((self.out_size, self.out_size), prefix + "W_a1" + layer_id)
        self.W_a2 = init_weights((self.out_size, self.out_size), prefix + "W_a2" + layer_id)
        self.W_a3 = init_weights((self.out_size, self.out_size), prefix + "W_a3" + layer_id)
        self.W_a4 = init_weights((self.out_size, self.out_size), prefix + "W_a4" + layer_id)
        self.U_a = init_weights((self.out_size, self.num_sents), prefix + "U_a" + layer_id)

        strength = T.dot(T.nnet.sigmoid(T.dot(sent_decs, self.W_a1) + T.dot(sent_encs, self.W_a2)), self.U_a)
        a = T.nnet.softmax(strength)
        c = T.dot(a, sent_encs)
        self.activation = T.tanh(T.dot(sent_decs, self.W_a3) + T.dot(c, self.W_a4))

        self.params = [self.W_a1, self.W_a2, self.W_a3, self.W_a4, self.U_a]

