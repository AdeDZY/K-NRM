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
