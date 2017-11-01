# K-NRM
---
This is the impelmentation of the paper [End-to-End Neural Ad-hoc Ranking with Kernel Pooling](http://www.cs.cmu.edu/~zhuyund/papers/end-end-neural.pdf) SIGIR17.

### Requirements
---
Tensorflow 0.12.1

### Guide To Use
---
First, configure the model through the config file. 

[Sample config](https://github.com/AdeDZY/KNRM/blob/master/sogou.knrm.config)

Then, pass the config file, training and validation data through the command:

[Sample shell scripts](https://github.com/AdeDZY/KNRM/blob/master/train-sogou-knrm.sh)

### Data Format
---
1. All queries and documents should be hashed into sequences of integer term ids.
2. Each training sample is a tuple of (query, postive document, negative documents)
3. Each testing sample is a tuple of (query, document)


Training:

query   \t postive_document   \t negative_document  \t score_difference 

177,705,632   \t  177,705,632,-1,2452,6,98   \t  177,705,632,3,25,14,37,2,146,159, -1   \t    0.119048

Testing:

q           tab document

177,705,632  \t   177,705,632,-1,2452,6,98

-1 indicate OOV or non-existence.


### Configurations 
---
TBA






