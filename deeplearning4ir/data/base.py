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
