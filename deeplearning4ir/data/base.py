# Copyright (c) 2017, Carnegie Mellon University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
base functions
"""
import logging
import numpy as np


def pair_docno(v_label, l_qid, l_docno):
    """
    make docno pairs (in position)
    :param v_label:
    :param l_qid:
    :param l_docno:
    :return: v_paired_label, l_paired_qid, l_docno_pair, l_pos_pair
    """
    logging.info('start pair documents...')
    v_paired_label = []
    l_paired_qid = []
    l_docno_pair = []
    l_pos_pair = []
    for p in xrange(len(l_qid)):
        for q in xrange(p + 1, len(l_qid)):
            if l_qid[q] != l_qid[p]:
                break
            if v_label[p] == v_label[q]:
                continue
            if v_label[p] > v_label[q]:
                label = 1.0
            else:
                label = -1.0
            l_pos_pair.append((p, q))
            l_paired_qid.append(l_qid[p])
            l_docno_pair.append((l_docno[p], l_docno[q]))
            v_paired_label.append(label)
    v_paired_label = np.array(v_paired_label)
    logging.info('total formed %d doc pairs', len(l_pos_pair))
    return v_paired_label, l_paired_qid, l_docno_pair, l_pos_pair


def fix_kfold_partition(with_dev=False, k=10, st=1, ed=100):
    l_train_folds = []
    l_dev_folds = []
    l_test_folds = []
    for fold in xrange(k):
        test = []
        train = []
        dev = []
        for qid in xrange(st, ed + 1):
            if (qid % k) == fold:
                test.append("%d" % qid)
                continue
            if with_dev:
                if ((qid + 1) % k) == fold:
                    dev.append("%d" % qid)
                    continue
            train.append("%d" % qid)
        l_train_folds.append(train)
        l_test_folds.append(test)
        l_dev_folds.append(dev)
    return l_train_folds, l_test_folds, l_dev_folds


def filter_svm_data(l_svm_data, l_qid):
    s = set(l_qid)
    return [svm_data for svm_data in l_svm_data if svm_data['qid'] in s]
