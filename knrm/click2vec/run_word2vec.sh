input=$1
emb_out=$2
./word2vec -train ${input} -output ${emb_out} -size 300 -window 2 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 0 -threads 16 -min-count 5