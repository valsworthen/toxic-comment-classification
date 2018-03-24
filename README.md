# Toxic Comment Classification  

This repo contains code for the [Kaggle Competition: Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)  

Currently it can be used to train different architectures I tried during the challenge. The best one (gru_cnn) achieves **AUC 0.9863** on Public leaderboard, and 0.9856 on Private Leaderboard.  

There are 5 different architectures, find out more [below](#models).  

# How to run  
## Requirements
Make sure you have Tensorflow **1.5** installed on GPU.

Input data should be inside `input/` folder and can be downloaded [here](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)  

This solution uses pre-trained embeddings so you should download a *pre-trained vectors* file ([see below](#embeddings) if you're not familiar with pre-trained embeddings)    

The script saves predictions & models after each fold and OOF predictions to do stacking.  

## Instructions  
### Quick run   

The *main* file is `train_nn.py` and can be run directly via command line specifying the path to your embedding file.  

For instance: `python3 train_nn.py crawl-300d-2M.vec`  

It is possible either to fit the model on 90% of the dataset or to perform K-fold cross validation.  

At the moment, `models.py` only contains the architectures mentioned below. Feel free to complete it with your own models!    

### Run with best configs  

Simply run `python3 train_nn.py crawl-300d-2M.vec --model model_name` where `model_name` is one of the following:  
- bibigru  
- pooled_gru  
- ngram_cnn  
- cnn_gru  
- gru_cnn  

For instance: `python3 train_nn.py crawl-300d-2M.vec --model bibigru`

# Appendix
## Models  

TBD: list the different models I used during the competitions  

## Embeddings  

Word embeddings are used in NLP to compute *vector representation* of words that can be fed to machine learning models. Such representations can be extracted from statistics of the corpus but it is now common to *learn* them via a "fake task". You can check [this great explanation of Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/), one of the most famous algorithm to learn word embeddings.  

However, learning precise representations requires **a lot** of training data and computing power. That's why it is common to download and use **pre-trained embeddings** such as:
- [GloVe](https://nlp.stanford.edu/projects/glove/)  
- [FastText](https://fasttext.cc/docs/en/english-vectors.html)  

I personally tried to use `crawl-300d-2M.vec` (embedding dimension=300) from fastText, and `glove.twitter.27B` (embedding dimension=200) and `glove.840B.300d` (embedding dimension=300) from GloVe.
