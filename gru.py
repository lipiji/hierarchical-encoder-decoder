#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class GRULayer(object):
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        prefix = "GRU_"
        layer_id = "_" + layer_id
        self.in_size, self.out_size = shape

        self.W_xr = init_weights((self.in_size, self.out_size), prefix + "W_xr" + layer_id)
        self.W_hr = init_weights((self.out_size, self.out_size), prefix + "W_hr" + layer_id)
        self.b_r = init_bias(self.out_size, prefix + "b_r" + layer_id)
        
        self.W_xz = init_weights((self.in_size, self.out_size), prefix + "W_xz" + layer_id)
        self.W_hz = init_weights((self.out_size, self.out_size), prefix + "W_hz" + layer_id)
        self.b_z = init_bias(self.out_size, prefix + "b_z" + layer_id)

        self.W_xh = init_weights((self.in_size, self.out_size), prefix + "W_xh" + layer_id)
        self.W_hh = init_weights((self.out_size, self.out_size), prefix + "W_hh" + layer_id)
        self.b_h = init_bias(self.out_size, prefix + "b_h" + layer_id)

        self.X = X
        self.M = mask
       
        def _active_mask(x, m, pre_h):
            x = T.reshape(x, (batch_size, self.in_size))
            pre_h = T.reshape(pre_h, (batch_size, self.out_size))

            r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
            z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
            gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
            h = (1 - z) * pre_h + z * gh
            
            h = h * m[:, None] + (1 - m[:, None]) * pre_h
            
            h = T.reshape(h, (1, batch_size * self.out_size))
            return h
        h, updates = theano.scan(_active_mask, sequences = [self.X, self.M],
                                 outputs_info = [T.alloc(floatX(0.), 1, batch_size * self.out_size)])
        # dic to matrix 
        h = T.reshape(h, (self.X.shape[0], batch_size * self.out_size))
        
        # dropout
        if p > 0:
            srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
            drop_mask = srng.binomial(n = 1, p = 1-p, size = h.shape, dtype = theano.config.floatX)
            self.activation = T.switch(T.eq(is_train, 1), h * drop_mask, h * (1 - p))
        else:
            self.activation = T.switch(T.eq(is_train, 1), h, h)
       
        self.params = [self.W_xr, self.W_hr, self.b_r,
                       self.W_xz, self.W_hz, self.b_z,
                       self.W_xh, self.W_hh, self.b_h]
    
    def _active(self, x, pre_h):
        r = T.nnet.sigmoid(T.dot(x, self.W_xr) + T.dot(pre_h, self.W_hr) + self.b_r)
        z = T.nnet.sigmoid(T.dot(x, self.W_xz) + T.dot(pre_h, self.W_hz) + self.b_z)
        gh = T.tanh(T.dot(x, self.W_xh) + T.dot(r * pre_h, self.W_hh) + self.b_h)
        h = z * pre_h + (1 - z) * gh
        return h


class BdGRU(object):
    # Bidirectional GRU Layer.
    def __init__(self, rng, layer_id, shape, X, mask, is_train = 1, batch_size = 1, p = 0.5):
        fwd = GRULayer(rng, "_fwd_" + layer_id, shape, X, mask, is_train, batch_size, p)
        bwd = GRULayer(rng, "_bwd_" + layer_id, shape, X[::-1], mask[::-1], is_train, batch_size, p)
        self.params = fwd.params + bwd.params
        self.activation = T.concatenate([fwd.activation, bwd.activation[::-1]], axis=1)

