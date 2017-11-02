# K-NRM
---
This is the implementation of the paper [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf).

### Requirements
---
- Tensorflow 0.12
- Numpy
- traitlets

### Guide To Use
---
First, configure the model through the config file. Configurable parameter are listed [here](#configurations)

[Sample config](https://github.com/AdeDZY/KNRM/blob/master/sogou.knrm.config)

**Training**: pass the config file, training data and validation data as
```ruby
python ./deeplearning4ir/clicknn/knrm.py config-file\
    --train \
    --train_file: path to training data\
    --validation_file: path to validation data\
    --train_size: size of training data (number of training samples)\
    --checkpoint_dir: directory to store/load model checkpoints\ 
    --load_model: True or False. Start with a new model or continue training
```

[Sample shell scripts](https://github.com/AdeDZY/KNRM/blob/master/train-sogou-knrm.sh)

**Testing**: pass the config file and testing data as
```ruby
python ./deeplearning4ir/clicknn/knrm.py config-file\
    --test \
    --test_file: path to testing data\
    --test_size: size of testing data (number of testing samples)\
    --checkpoint_dir: directory to load trained model\
    --output_score_file: file to output documents score\

```
Relevance scores will be output to output_score_file, one score per line, in the same order as test_file.
We provide tools to convert score into trec format in.
```ruby
python ./tools/gen_trec_from_score
```

### Data Preperation
---
1. All queries and documents should be hashed into sequences of integer term ids. 
-1 indicates OOV or non-existence. Term ids are sepereated by `,`

2. Each training sample is a tuple of (query, postive document, negative documents)
3. Each testing sample is a tuple of (query, document)


**Training**

query   \t postive_document   \t negative_document  \t score_difference 

177,705,632   \t  177,705,632,-1,2452,6,98   \t  177,705,632,3,25,14,37,2,146,159, -1   \t    0.119048

**Testing**

q   \t document

177,705,632  \t   177,705,632,-1,2452,6,98



### Configurations 
---

**Model Configurations**
- <code>ClickNN.n_bins</code>: number of kernels (soft bins) (Default: 11)
- <code>KNRM.lamb</code>: defines the guassian kernels' sigma value. sigma = lamb * bin_size (Default:0.5 -> sigma=0.1)
- <code>ClickNN.embedding_size</code>: embedding dimension (Default: 300)
- <code>ClickNN.max_q_len</code>: max query length (Default: 10)
- <code>ClickNN.max_d_len</code>: max document length (Default: 50)
- <code>ClickDataGenerator.max_q_len</code>: max query length. Should be the same as <code>ClickNN.max_q_len</code> (Default: 10)
- <code>ClickDataGenerator.max_d_len</code>: max query length. Should be the same as <code>ClickNN.max_d_len</code> (Default: 50)
- <code>ClickNN.vocabulary_size</code>: vocabulary size.
- <code>ClickDataGenerator.vocabulary_size</code>: vocabulary size.



**Data**
- <code>KNRM.emb_in</code>: initial embeddings
- <code>ClickDataGenerator.min_score_diff</code>: 
minimum score differences between postive documents and negative ones (default=0)

**Training Parameters**
- <code>ClickNN.bath_size</code>: batch size. (Default: 16)
- <code>ClickNN.max_epochs</code>: max number of epochs to train
- <code>ClickNN.eval_frequency</code>: evaluate model on validation set very this steps (Default: 10000)
- <code>ClickNN.checkpoint_steps</code>: save model very this steps (Default: 10000)
- <code>KNRM.learning_rate</code>: learning rate for Adam Opitmizer (Default: 0.001)
- <code>KNRM.epsilon</code>: epsilon for Adam Optimizer (Default: 0.00001)

---
If you use this code for academic purposes, please cite it as:

```
@inproceedings{xiong2017neural,
  author          = {{Xiong}, Chenyan and {Dai}, Zhuyun and {Callan}, Jamie and {Liu}, Zhiyuan and {Power}, Russell},
  title           = "{End-to-End Neural Ad-hoc Ranking with Kernel Pooling}",
  booktitle       = {Proceedings of the 40th International ACM SIGIR Conference on Research & Development in Information Retrieval},
  organization    = {Association for Computational Linguistics},
  year            = 2017,
}
```




