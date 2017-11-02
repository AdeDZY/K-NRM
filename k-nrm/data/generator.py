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
ranking data generator
include:
    point wise generator, each time yield a:
        query-doc pair (x), y
        x is a dict: h['q'] = array of q (batch size * max q len)
                    h['d'] = array of d (batch size * max title len)
                    h['idf] = None or array of query term idf (batch size * max q len)

    pair wise generator, each time yield a
        query-doc+, doc- pair (x), y
            the same x with pointwise's, with an additional key:
                h['d_aux'] = array of the second d (batch size * max title len)

    all terms are int ids


"""


from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Int,
    List,
    Float,
    Bool,
)
import numpy as np
from numpy import genfromtxt
import logging
from io import StringIO
import sys
reload(sys)
sys.setdefaultencoding('UTF8')


class ClickDataGenerator(Configurable):
    title_in = Unicode('/bos/data1/sogou16/data/training/1m_title.pad_t50',
                       help='titles term id csv, must be padded').tag(config=True)
    max_q_len = Int(10, help='max q len').tag(config=True)
    max_d_len = Int(50, help='max document len').tag(config=True)
    q_name = Unicode('q')
    d_name = Unicode('d')
    aux_d_name = Unicode('d_aux')
    idf_name = Unicode('idf')
    neg_sample = Int(1, help='negative sample').tag(config=True)
    load_litle_pool=Bool(False, help='load little pool at beginning').tag(conf=True)
    min_score_diff = Float(0, help='min score difference for click data generated pairs').tag(config=True)
    vocabulary_size = Int(2000000).tag(config=True)

    def __init__(self, **kwargs):
        super(ClickDataGenerator, self).__init__(**kwargs)
        self.m_title_pool = np.array(None)
        if self.load_litle_pool and self.neg_sample:
            self._load_title_pool()
        print "min_score_diff: ", self.min_score_diff
        print "generator's vocabulary size: ", self.vocabulary_size

    def _load_title_pool(self):
        if self.title_in:
            logging.info('start loading title pool [%s]', self.title_in)
            self.m_title_pool = genfromtxt(self.title_in, delimiter=',',  dtype=int,)
            logging.info('loaded [%d] title pool', self.m_title_pool.shape[0])

    def pointwise_generate(self, pair_stream, batch_size, with_label=True, with_idf=False):
        """
        to use: initial the generator = ClickDataGenerator(config=conf)
            and then for X,Y in generator.pointwise_generator(pair_stream, batch_size)
        :param pair_stream: the (probably infinite) stream of query \t clicked url
            e.g. itertools.cycle(open(file))
        :param batch_size: int, a batch size
        :param with_label: if True, then there is a third column in pair_stream: \t label (int)
        :param with_idf: if True, the third col in pair_stream is the query term idf
        :return: yield a batched X and Y
        """
        l_q = []
        l_d = []
        l_idf = []
        l_y = []
        for line in pair_stream:
            cols = line.split('\t')
            q = np.array([int(t) for t in cols[0].split(',')])  #np.loadtxt(StringIO(unicode(cols[0])), delimiter=',',  dtype=int,)
            doc = np.array([int(t) for t in cols[1].split(',')])  #np.loadtxt(StringIO(unicode(cols[1])), delimiter=',',  dtype=int,)
            
            if with_idf:
                idf = np.array([int(t) for t in cols[2].split(',')])
            y = 0
            if with_label:
                y = int(cols[-1])
            v_q = np.ones(self.max_q_len) * -1
            v_d = np.ones(self.max_d_len) * -1
            v_q[:min(q.shape[0], self.max_q_len)] = q[0:min(q.shape[0], self.max_q_len)]
            v_d[:min(doc.shape[0], self.max_d_len)] = doc[0:min(doc.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_y.append(y)

            if with_idf:
                v_idf = np.zeros(self.max_q_len)
                v_idf[:idf.shape[0]] = idf[0:min(q.shape[0], self.max_q_len)]
                l_idf.append(v_idf)

            if len(l_q) >= batch_size:
                Q = np.array(l_q,  dtype=int,)
                D = np.array(l_d,  dtype=int,)
                IDF = None
                if with_idf:
                    IDF = np.array(l_idf, dtype=float)
                Y = np.array(l_y,  dtype=int,)
                X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF}
                yield X, Y
                l_q, l_d, l_y, l_idf = [], [], [], []
        if l_q:
            Q = np.array(l_q,  dtype=int,)
            D = np.array(l_d,  dtype=int,)
            IDF = None
            if with_idf:
                IDF = np.array(l_idf, dtype=float)
            Y = np.array(l_y,  dtype=int,)
            X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF}
            yield X, Y
        logging.info('point wise generator to an end')

    def pairwise_generate(self, pair_stream, batch_size, with_idf=False):
        """
        to use: initial the generator = ClickDataGenerator(config=conf)
            and then for X,Y in generator.pairwise_generate(pair_stream, batch_size)
        :param pair_stream: the (probably infinite) stream of query \t clicked url
            e.g. itertools.cycle(open(file))
        :param batch_size: must be neg_sample * k size
        :param with_idf: if True, the third col in pair_stream is the query term idf
        :return: yield a batched X and Y
            NOTE: the Y is always 1, the order of pos and neg docs are not shuffled yet.
        """

        assert batch_size % self.neg_sample == 0
        pos_batch_size = batch_size / self.neg_sample

        for pos_X, pos_Y in self.pointwise_generate(pair_stream, pos_batch_size, with_label=False, with_idf=with_idf):
            idx = np.random.randint(self.m_title_pool.shape[0], size=batch_size)
            aux_D = self.m_title_pool[idx, :]
            new_Q = np.repeat(pos_X[self.q_name], self.neg_sample, axis=0)
            new_D = np.repeat(pos_X[self.d_name], self.neg_sample, axis=0)

            new_IDF = None
            if with_idf:
                new_IDF = np.repeat(pos_X[self.idf_name], self.neg_sample, axis=0)

            X = {self.q_name: new_Q, self.d_name: new_D, self.aux_d_name: aux_D, self.idf_name:new_IDF}
            Y = np.ones(batch_size)
            yield X, Y

    def pairwise_reader(self, pair_stream, batch_size, with_idf=False):
        l_q = []
        l_d = []
        l_d_aux = []
        l_idf = []
        l_y = []
        for line in pair_stream:
            cols = line.strip().split('\t')
            if len(cols) < 4: continue
            flag = True
            for col in cols:
                if not col.strip():
                    flag = False
                    break
            if not flag:
                print line
                continue
            y = float(cols[3])
            if abs(y) < self.min_score_diff:
                continue
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocabulary_size])
            t1 = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocabulary_size])
            t2 = np.array([int(t) for t in cols[2].split(',') if int(t) < self.vocabulary_size])
            if y > 0:
                y = 1
            else:
                t1, t2 = t2, t1  # make the first always positive
                y = 1
            if with_idf:
                if len(cols) < 5:
                    idf = np.ones(len(q))
                else:
                    idf = np.array([float(t) for t in cols[4].split(',')])
            v_q = np.ones(self.max_q_len) * -1
            v_d = np.ones(self.max_d_len) * -1
            v_d_aux = np.ones(self.max_d_len) * -1
            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t1.shape[0], self.max_d_len)] = t1[:min(t1.shape[0], self.max_d_len)]
            v_d_aux[:min(t2.shape[0], self.max_d_len)] = t2[:min(t2.shape[0], self.max_d_len)]

            l_q.append(v_q)
            l_d.append(v_d)
            l_d_aux.append(v_d_aux)
            l_y.append(y)

            if with_idf:
                v_idf = np.zeros(self.max_q_len)
                v_idf[:min(idf.shape[0], self.max_q_len)] = idf[:min(idf.shape[0], self.max_q_len)]
                l_idf.append(v_idf)

            if len(l_q) >= batch_size:
                Q = np.array(l_q,  dtype=int,)
                D = np.array(l_d,  dtype=int,)
                D_aux = np.array(l_d_aux, dtype=int)
                IDF = None
                if with_idf:
                    IDF = np.array(l_idf, dtype=float)
                Y = np.array(l_y,  dtype=int,)
                X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF, self.aux_d_name: D_aux}
                yield X, Y
                l_q, l_d, l_d_aux, l_y, l_idf = [], [], [], [], []
        if l_q:
            Q = np.array(l_q,  dtype=int,)
            D = np.array(l_d,  dtype=int,)
            D_aux = np.array(l_d_aux,  dtype=int,)
            IDF = None
            if with_idf:
                IDF = np.array(l_idf, dtype=float)
            Y = np.array(l_y,  dtype=int,)
            X = {self.q_name: Q, self.d_name: D, self.idf_name: IDF, self.aux_d_name: D_aux}
            yield X, Y
        logging.info('pair wise reader to an end')

if __name__ == '__main__':
    from deeplearning4ir.utils import set_basic_log, load_py_config
    set_basic_log()
    if 4 != len(sys.argv):
        print "I test generator"
        print "3 para: config + click pair with int term + batch_size"
        ClickDataGenerator.class_print_help()
        sys.exit(-1)
    conf = load_py_config(sys.argv[1])
    generator = ClickDataGenerator(config=conf)

    pair_stream = open(sys.argv[2])
    batch_size = int(sys.argv[3])
    X, Y = next(generator.pointwise_generate(pair_stream, batch_size, False))
    a = np.ones(1)

    print "point wise Y:\n %s" % (np.array2string(Y))
    print "\n\n"
    print "q: \n %s" % (np.array2string(X[generator.q_name]))
    print "\n\n"
    print "d: \n %s" % (np.array2string(X[generator.d_name]))
    print "\n\n"
    X, Y = next(generator.pairwise_generate(pair_stream, batch_size))

    print "pair wise Y:\n %s" % (np.array2string(Y))
    print "\n\n"
    print "q: \n %s" % (np.array2string(X[generator.q_name]))
    print "\n\n"
    print "d: \n %s" % (np.array2string(X[generator.d_name]))
    print "\n\n"
    print "aux d: \n %s" % (np.array2string(X[generator.aux_d_name]))













