"""
generate click term pair from query-clicked doc title:
    one term from query, one term from a clicked doc title
input:
    a file a query \t url \t title
output:
    click term pair
    one pair a line
"""

import sys
reload(sys)
sys.setdefaultencoding('UTF8')


if len(sys.argv) != 3:
    print "I generate click pair"
    print "2 para: q-clicked url -title file + output"
    sys.exit()


pair_cnt = 0
out = open(sys.argv[2], 'w')

for line_cnt, line in enumerate(open(sys.argv[1])):
    cols = line.strip().split('\t')
    q, t = cols[-2:]
    # url, q, t = line.strip().split('\t')
    for qt in q.split():
        for tt in t.split():
            print >> out, qt + '\t' + tt
            pair_cnt += 1
    if not line_cnt % 1000:
        print "read [%d] clicks [%d] term pair generated" % (line_cnt, pair_cnt)

print "finished with [%d] lines [%d] pairs" % (line_cnt + 1, pair_cnt + 1)
