# Toxic Comment Classification  

This repo contains code for the [Kaggle Competition: Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  

Currently it can be used to train different architectures I tried during the challenge. The best one (gru_cnn) achieves **AUC 0.9863** on Public leaderboard, and 0.9856 on Private Leaderboard.  

There are 5 different architectures, find out more [below](#models).  

# How to run  
## Requirements
Make sure you have Tensorflow **1.5** installed on GPU.  
Run: `pip install scikit-learn keras tqdm`

Input data should be inside `input/` folder and can be downloaded [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). I've only put sample files inside `input/` so that I can push the folder and perform tests.  

This solution uses pre-trained embeddings so you should download a *pre-trained vectors* file ([see below](#embeddings) if you're not familiar with pre-trained embeddings)    

The script saves predictions & models after each fold and OOF predictions to do stacking.  

## Instructions  
### Quick run   
tl;dr `python3 train_nn.py`

The *main* file is `train_nn.py` and can be run directly via command line specifying the path to your embedding file.  

For instance: `python3 train_nn.py crawl-300d-2M.vec` (you can also run it without argument, and it will use a sample of fastText vectors!)  

Running `train_nn.py` reads parameters from `parameters.yaml`:  
In particulier `MODEL_TYPE` picks from `models.py` one model that will be trained. Therefore you should modify `models.py` whenever you want to try a new architecture.
See next part to know how to run the models I tried during the competition with fine-tuned parameters!   

### Run with best configs  

Simply run `python3 train_nn.py crawl-300d-2M.vec --model model_name` where `model_name` is one of the following:  
- [bibigru](#bibigru)  
- [pooled_gru](#pooledgru)  
- [ngram_cnn](#n-gram-cnn)  
- [cnn_gru](#cnn-gru)  
- [gru_cnn](#gru-cnn)  

For instance: `python3 train_nn.py crawl-300d-2M.vec --model bibigru`

# Appendix
## Models  

I mostly tried GRU and CNN models during the competition. Almost every models starts with a non-trainable embedding layer followed by SpatialDropout and ends with a 6-dimensional Dense sigmoid layer so I won't mention it later.  

#### BiBiGRU
This model was inspired by [Deepmoji](https://github.com/bfelbo/DeepMoji). It scored 0.9853 in Public LB.
It consists in two successive Bidirectional GRU layers, whose outputs are then concatenated with the output of the SpatialDropout layer (so this is almost the original representation of the commentary).  
Next layer is Attention (could be Max/Average pooling) followed by a 50-Dense layer.  

#### PooledGRU  
This is a simpler GRU model which was proposed in [Kaggle Kernel by Vladimir Demidov](https://www.kaggle.com/yekenot/pooled-gru-fasttext). It scored 0.9859 in Public LB.  
There is only one GRU layer followed by Average and Max pooling. The outputs of the GRU, Max pooling and Average pooling are then concatenated.  

#### N-gram CNN
This model was inspired by [a discussion](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/50501) about the performance of CNNs compared to RNNs. It scored 0,9849 in Public LB.  
The objective of the architecture is to capture bigrams, trigrams, etc. inside the commentaries. Therefore, several conv layers with **different kernel sizes** are *simultaneously* trained on the ouput of the SpatialDropout layer. All the conv layers are followed by Attention/MaxPooling/AveragePooling and then concatenated before being fed to an optional 50-Dense layer.  

#### CNN GRU  
This model scored 0.9861 in Public LB.  
The name of this architecture is not really correct because the GRU layer **does not** follow the convolutional blocks. It is rather trained directly on the sequences and its output is concatenated with the outputs of the conv blocks.  
In the end this model is very similar with ngram CNN but with a GRU layer trained in parallel.  

#### GRU CNN
This is the best performing architecture but also the heaviest. It scored 0.9863 in Public LB.  
It is a mix of BiBiGRU and ngram CNN. The sequences are fed to 2 successives BiGRU layers whose outputs are concatenated. On top of that, the convolution blocks of ngram CNN are trained.  

## Embeddings  

Word embeddings are used in NLP to compute *vector representation* of words that can be fed to machine learning models. Such representations can be extracted from statistics of the corpus but it is now common to *learn* them via a "fake task". You can check [this great explanation of Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/), one of the most famous algorithm to learn word embeddings.  

However, learning precise representations requires **a lot** of training data and computing power. That's why it is common to download and use **pre-trained embeddings** such as:
- [GloVe](https://nlp.stanford.edu/projects/glove/)  
- [FastText](https://fasttext.cc/docs/en/english-vectors.html)  

I personally tried to use `crawl-300d-2M.vec` (embedding dimension=300) from fastText, and `glove.twitter.27B` (embedding dimension=200) and `glove.840B.300d` (embedding dimension=300) from GloVe.
