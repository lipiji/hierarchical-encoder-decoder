# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

def load_stop_words(f_path = None):
    stop_words = {}
    if f_path == None:
        f = open(curr_path + "/data/stopwords.txt", "r")
    else:
        f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        stop_words[line] = 1

    return stop_words

def char_sequence(f_path = None, batch_size = 1):
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    if f_path == None:
        f = open(curr_path + "/data/toy.txt", "r")
    else:
        f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        if len(line) < 3:
            continue
        lines.append(line)
        for char in line:
            if char not in w2i:
                i2w[len(w2i)] = char
                w2i[char] = len(w2i)
    f.close()

    for i in range(0, len(lines)):
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = batch_sequences(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

def word_sequence(f_path, batch_size = 1):
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    tf = {}
    f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        words = line.split()
        if len(words) < 3 or line == "====":
            continue
        lines.append(words)
        for w in words:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
                tf[w] = 1
            else:
                tf[w] += 1
    f.close()

    for i in range(0, len(lines)):
        words = lines[i]
        x = np.zeros((len(words), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(words)):
            x[j, w2i[words[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = batch_sequences(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

def batch_sequences(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            max_len = np.max(seqs_len);
            mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            
            concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)
            concat_Y = concat_X.copy()
            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                Y = batch_y[b_i]
                mask[0 : X.shape[0], b_i] = 1
                for r in xrange(max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                    Y = np.concatenate((Y, zeros_m), axis=0)
                concat_X[:, b_i * dim : (b_i + 1) * dim] = X 
                concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y
            data_xy[batch_id] = [concat_X, concat_Y, mask, len(batch_x)]
            batch_x = []
            batch_y = []
            seqs_len = []
            batch_id += 1
    return data_xy

