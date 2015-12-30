#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data

use_gpu(0) # -1:cpu; 0,1,2,..: gpu

e = 0.0
lr = 0.5
drop_rate = 0.
batch_size = 1000
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 

seqs, i2w, w2i, data_xy = data.word_sequence("/data/toy.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i)
num_sents = data_xy[0][3]
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)

print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(2000):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0]
        mask = xy[2]
        local_batch_size = xy[3]
        cost, sents = model.train(X, mask, lr, local_batch_size)
        error += cost
        #print i, g_error, (batch_id + 1), "/", len(data_xy), cost
    in_time = time.time() - in_start

    for s in xrange(int(sents.shape[1] / dim_y)):
        xs = sents[:, s * dim_y : (s + 1) * dim_y]
        for w in xrange(xs.shape[0]):
            if i2w[np.argmax(xs[w,:])] == "<eoss>":
                break
            print i2w[np.argmax(xs[w,:])],
        print "\n"

    error /= len(data_xy);
    if error < g_error:
        g_error = error

    print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/char_rnn.model", model)

