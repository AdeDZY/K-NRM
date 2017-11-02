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

import argparse
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("score_file", help="K-NRM output score file", type=argparse.FileType('r'))
    parser.add_argument("trec_org_file", help="must in the same order as score_file", type=argparse.FileType('r'))
    parser.add_argument("out_file", type=argparse.FileType('w'))
    args = parser.parse_args()

    # read score and trec format
    res = []
    for score, trec_line in zip(args.score_file, args.trec_org_file):
        score = float(score)
        items = trec_line.strip().split('\t')
        qid = items[0]
        docid = items[2]

        res.append((qid, score, docid))

    res = sorted(res, reverse=True)
    prev_qid = -1
    rank = 0
    for qid, score, docid in res:
        if prev_qid != qid:
            prev_qid = qid
            rank = 0
        rank += 1
        args.out_file.write("{0}\tQ0\t{1}\t{2}\t{3}\tknrm\n".format(qid, docid, rank, score))


