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
generate top_n term pair from trec format:
    one term from query, one term from a top_n doc title and body
input:
    a file a query \t url \t title
output:
    term pair
    one pair a line
"""

import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file_with_info", type=argparse.FileType('r'))
    parser.add_argument("vocab_file", type=argparse.FileType('r'))
    parser.add_argument("output_file", type=argparse.FileType('w'))
    parser.add_argument("--topn", '-n', type=int, default=10)
    args = parser.parse_args()

    # read vocab
    vocab = set()
    for line in args.vocab_file:
        term, tid = line.split('\t')
        vocab.add(term)
    print "Vocabuloray size: {0}.".format(len(vocab))

    prev_qid = -1
    rank = 0
    for line in args.trec_file_with_info:
        first_lvl = line.strip().split('#')
        data = '#'.join(first_lvl[1:])
        cols = first_lvl[0].split('\t')
        qid = int(cols[0])
        if qid != prev_qid:
            prev_qid = qid
            rank = 0
        rank += 1
        if rank > args.topn:
            continue
        h = json.loads(data)

        query = h['query']
        title = h['doc']['title'].strip()
        body = h['doc'].get('body', '').strip()

        query_terms = query.split(' ')
        doc_terms = title.split(' ') + body.split(' ')
        for term1 in query_terms:
            if term1 not in vocab:
                continue
            for term2 in doc_terms:
                if term2 not in vocab:
                    continue
                args.output_file.write('{0}\t{1}\n'.format(term1, term2))



