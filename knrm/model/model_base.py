# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE

import tensorflow as tf
import numpy as np
from knrm.data import DataGenerator
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
)

import sys

reload(sys)
sys.setdefaultencoding('UTF8')


class BaseNN(Configurable):
    n_bins = Int(11, help="number of kernels (including exact match)").tag(config=True)
    max_q_len = Int(10, help='max q len').tag(config=True)
    max_d_len = Int(50, help='max document len').tag(config=True)
    batch_size = Int(16, help="minibatch size").tag(config=True)
    max_epochs = Float(10, help="maximum number of epochs").tag(config=True)
    eval_frequency = Int(10000, help="print out minibatch every * steps").tag(config=True)
    checkpoint_steps = Int(10000, help="stroe trained data every * steps").tag(config=True)
    embedding_size = Int(300, help="embedding dimension").tag(config=True)
    vocabulary_size = Int(2000000, help="vocabulary size").tag(config=True)

    def __init__(self, **kwargs):
        super(BaseNN, self).__init__(**kwargs)

        # generator
        self.data_generator = DataGenerator(config=self.config)
        self.val_data_generator = DataGenerator(config=self.config)
        self.test_data_generator = DataGenerator(config=self.config)

    @staticmethod
    def kernal_mus(n_kernels, use_exact):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in xrange(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    @staticmethod
    def kernel_sigmas(n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma

    @staticmethod
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial)

    @staticmethod
    def re_pad(D, batch_size):
        D = np.array(D)
        D[D < 0] = 0
        if len(D) < batch_size:
            tmp = np.zeros((batch_size - len(D), D.shape[1]))
            D = np.concatenate((D, tmp), axis=0)
        return D

    def gen_mask(self, Q, D, use_exact=True):
        """
        Generate mask for the batch. Mask padding and OOV terms.
        Exact matches is alos masked if use_exat == False.
        :param Q: a batch of queries, [batch_size, max_len_q]
        :param D: a bacth of documents, [batch_size, max_len_d]
        :param use_exact: mask exact matches if set False.
        :return: a mask of shape [batch_size, max_len_q, max_len_d].
        """
        mask = np.zeros((self.batch_size, self.max_q_len, self.max_d_len))
        for b in range(len(Q)):
            for q in range(len(Q[b])):
                if Q[b, q] > 0:
                    mask[b, q, D[b] > 0] = 1
                    if not use_exact:
                        mask[b, q, D[b] == Q[b, q]] = 0
        return mask


