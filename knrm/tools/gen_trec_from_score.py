# Copyright (c) 2017, Carnegie Mellon University. All rights reserved.
#
# Use of the K-NRM package is subject to the terms of the software license set
# forth in the LICENSE file included with this software, and also available at
# https://github.com/AdeDZY/K-NRM/blob/master/LICENSE

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


