# K-NRM
This is the implementation of the Kernel-based Neural Ranking Model (K-NRM) model from paper [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf).

<p align="center"> 
<img src="https://github.com/AdeDZY/K-NRM/blob/master/model_simplified-1.png" width="400" align="center">
</p>

If you use this code for your scientific work, please cite it as ([bibtex](#cite-the-paper)):

```
C. Xiong, Z. Dai, J. Callan, Z. Liu, and R. Power. End-to-end neural ad-hoc ranking with kernel pooling. 
In Proceedings of the 40th International ACM SIGIR Conference on Research & Development in Information Retrieval. 
ACM. 2017.
```



### Requirements
---
- Tensorflow 0.12 
- Numpy
- traitlets

Coming soon: K-NRM with Tensorflow 1.0

### Guide To Use
---
**Configure**: first, configure the model through the config file. Configurable parameters are listed [here](#configurations)

[sample.config](https://github.com/AdeDZY/K-NRM/blob/master/sample.config)

**Training** : pass the config file, training data and validation data as
```ruby
python ./knrm/model/model_knrm.py config-file\
    --train \
    --train_file: path to training data\
    --validation_file: path to validation data\
    --train_size: size of training data (number of training samples)\
    --checkpoint_dir: directory to store/load model checkpoints\ 
    --load_model: True or False. Start with a new model or continue training
```

[sample-train.sh](https://github.com/AdeDZY/K-NRM/blob/master/sample-train.sh)

**Testing**: pass the config file and testing data as
```ruby
python ./knrm/model/model_knrm.py config-file\
    --test \
    --test_file: path to testing data\
    --test_size: size of testing data (number of testing samples)\
    --checkpoint_dir: directory to load trained model\
    --output_score_file: file to output documents score\

```
Relevance scores will be output to output_score_file, one score per line, in the same order as test_file.
We provide a script to convert scores into trec format.
```ruby
./knrm/tools/gen_trec_from_score.py
```

### Data Preperation
---
All queries and documents must be mapped into sequences of integer term ids. Term id starts with 1.
-1 indicates OOV or non-existence. Term ids are sepereated by `,`

**Training Data Format**

Each training sample is a tuple of (query, postive document, negative document)

`query   \t postive_document   \t negative_document  \t score_difference `

Example: `177,705,632   \t  177,705,632,-1,2452,6,98   \t  177,705,632,3,25,14,37,2,146,159, -1   \t    0.119048`

If `score_difference < 0`, the data generator will swap postive docment and negative document.

If `score_difference < lickDataGenerator.min_score_diff`, this training sample will be omitted.

We recommend shuffling the training samples to ease model convergence. 

**Testing Data Format**

Each testing sample is a tuple of (query, document)

`q   \t document`

Example: `177,705,632  \t   177,705,632,-1,2452,6,98`



### Configurations 
---

**Model Configurations**
- <code>BaseNN.n_bins</code>: number of kernels (soft bins) (default: 11. One exact match kernel and 10 soft kernels)
- <code>Knrm.lamb</code>: defines the guassian kernels' sigma value. sigma = lamb * bin_size (default:0.5 -> sigma=0.1)
- <code>BaseNN.embedding_size</code>: embedding dimension (default: 300)
- <code>BaseNN.max_q_len</code>: max query length (default: 10)
- <code>BaseNN.max_d_len</code>: max document length (default: 50)
- <code>DataGenerator.max_q_len</code>: max query length. Should be the same as <code>BaseNN.max_q_len</code> (default: 10)
- <code>DataGenerator.max_d_len</code>: max query length. Should be the same as <code>BaseNN.max_d_len</code> (default: 50)
- <code>BaseNN.vocabulary_size</code>: vocabulary size.
- <code>DataGenerator.vocabulary_size</code>: vocabulary size.



**Data**
- <code>Knrm.emb_in</code>: initial embeddings
- <code>DataGenerator.min_score_diff</code>: 
minimum score differences between postive documents and negative ones (default: 0)

**Training Parameters**
- <code>BaseNN.bath_size</code>: batch size (default: 16)
- <code>BaseNN.max_epochs</code>: max number of epochs to train
- <code>BaseNN.eval_frequency</code>: evaluate model on validation set very this steps (default: 1000)
- <code>BaseNN.checkpoint_steps</code>: save model very this steps (default: 10000)
- <code>Knrm.learning_rate</code>: learning rate for Adam Opitmizer (default: 0.001)
- <code>Knrm.epsilon</code>: epsilon for Adam Optimizer (default: 0.00001)

Efficiency
---
During training, it takes about 60ms to process one batch on a single-GPU machine with the following settings:
- batch size: 16
- max_q_len: 10
- max_d_len: 50
- vocabulary_size: 300K

Smaller vocabulary and shorter documents accelerate the training.

### Click2Vec
---
We also provide the click2vec model as described in our paper.
- <code>./knrm/click2vec/generate_click_term_pair.py</code>: generate <query_term, clicked_title_term> pairs
- <code>./knrm/click2vec/run_word2vec.sh</code>: call Google's word2vec tool to train click2vec.

### Cite the paper
---
If you use this code for your scientific work, please cite it as:

```
C. Xiong, Z. Dai, J. Callan, Z. Liu, and R. Power. End-to-end neural ad-hoc ranking with kernel pooling. 
In Proceedings of the 40th International ACM SIGIR Conference on Research & Development in Information Retrieval. 
ACM. 2017.
```

```
@inproceedings{xiong2017neural,
  author          = {{Xiong}, Chenyan and {Dai}, Zhuyun and {Callan}, Jamie and {Liu}, Zhiyuan and {Power}, Russell},
  title           = "{End-to-End Neural Ad-hoc Ranking with Kernel Pooling}",
  booktitle       = {Proceedings of the 40th International ACM SIGIR Conference on Research & Development in Information Retrieval},
  organization    = {ACM},
  year            = 2017,
}
```




