python ./k-nrm/model/knrm.py sogou.knrm.config --train \
      --train_file "../data1/train-dev/pad-title-url/train.DCTR.pairs.hashed.shuf" \
      --validation_file "../data1/train-dev/pad-title-url/dev.pairs.hashed.shuf" \
      --train_size 8553124 \
      --checkpoint_dir "./output/test/" 
