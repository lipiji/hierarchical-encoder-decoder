#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

# set use gpu programatically
import theano.sandbox.cuda
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size,))), name)

def rmse(py, y):
    e = 0
    for t in xrange(len(y)):
        e += np.sqrt(np.mean((np.asarray(py[t,]) - np.asarray(y[t,])) ** 2))
    return e / len(y)

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    return model
