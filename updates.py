#pylint: skip-file
#https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
import numpy as np
import theano
import theano.tensor as T

'''
def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(n >= c, g * c / n, g)
    return g

def clip(x, min_value, max_value):
    if max_value < min_value:
        max_value = min_value
    return T.clip(x, min_value, max_value)
'''

def sgd(params, gparams, learning_rate = 0.1):
    updates = []
    for p, g in zip(params, gparams):
        updates.append((p, p - learning_rate * g))
    return updates

def momentum(params, gparams, learning_rate = 0.1, momentum = 0.9):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        velocity = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        x = momentum * velocity - learning_rate * g
        updates.append((velocity, x))
        updates.append((p, p + x))
    return updates

def nesterov_momentum(params, gparams, learning_rate = 0.1, momentum = 0.9):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        velocity = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        x = momentum * velocity - learning_rate * g
        updates.append((velocity, x))
        inc = momentum * x - learning_rate * g
        updates.append((p, p + inc))
    return updates

def rmsprop(params, gparams, learning_rate = 0.001, rho = 0.9, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g / T.sqrt(acc_new + epsilon)))
    return updates

def adagrad(params, gparams, learning_rate = 0.01, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        acc_new = acc + g ** 2
        updates.append((acc, acc_new))
        updates.append((p, p - learning_rate * g / T.sqrt(acc_new + epsilon)))
    return updates

def adadelta(params, gparams, learning_rate = 1.0, rho = 0.95, epsilon = 1e-6):
    updates = []
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        delta_acc = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        
        acc_new = rho * acc + (1 - rho) * g ** 2
        updates.append((acc, acc_new))
        
        update = (g * T.sqrt(delta_acc + epsilon) / T.sqrt(acc_new + epsilon))
        updates.append((p, p - learning_rate * update))

        delta_acc_new = rho * delta_acc + (1 - rho) * update ** 2
        updates.append((delta_acc, delta_acc_new))
    return updates

def adam(params, gparams, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    updates = []
    t_pre = theano.shared(np.asarray(.0, dtype=theano.config.floatX))
    t = t_pre + 1
    a_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    for p, g in zip(params, gparams):
        v = p.get_value(borrow = True)
        m_pre = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        v_pre = theano.shared(np.zeros(v.shape, dtype = v.dtype), broadcastable = p.broadcastable)
        
        m_t = beta1 * m_pre + (1 - beta1) * g
        v_t = beta2 * v_pre + (1 - beta2) * g ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)
        
        updates.append((m_pre, m_t))
        updates.append((v_pre, v_t))
        updates.append((p, p - step))

    updates.append((t_pre, t))
    return updates
