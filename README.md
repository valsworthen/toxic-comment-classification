# How to run  
## Requirements
Make sure you have Tensorflow **1.5** installed on GPU.  

The script saves a lot of files (OOF, predictions, models) in the following folders:  
- `temporary_cv_models/`  
- `temporary_cv_preds/`  
- `temporary_oof_preds/`  

Input data should be inside `input/` folder.  

**Make sure to replace the** `embedding` **dictionnary in**`train_nn.py` **with your paths to pre-trained embeddings** ([see below](#embeddings) if you're not familiar with pre-trained embeddings)  

## Instructions  
### Run with best configs  

Simply run `python3 train_nn.py` + `model_name` from one of the following:  
- bibigru  
- pooled_gru  
- ngram_cnn  
- cnn_gru  
- gru_cnn  

For instance: `python3 train_nn.py bibigru`

### Run with custom configs  

The *main* file is `train_nn.py` and can be run directly via command line after having specified the parameters in `parameters.yaml`.  

Command: `python3 train_nn.py`  

It is possible either to fit the model on 90% of the dataset or to perform K-fold cross validation.  

At the moment, `models.py` only contains the architectures mentioned above. Feel free to complete it with your own models!    

# Appendix
## Models  

TBD: list the different models I used during the competitions  

## Embeddings  

Word embeddings are used in NLP to compute *vector representation* of words that can be fed to machine learning models. Such representations can be extracted from statistics of the corpus but it is now common to *learn* them via a "fake task". You can check [this great explanation of Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/), one of the most famous algorithm to learn word embeddings.  

However, learning precise representations requires **a lot** of training data and computing power. That's why it is common to download and use **pre-trained embeddings** such as:
- [GloVe](https://nlp.stanford.edu/projects/glove/)  
- [FastText](https://fasttext.cc/docs/en/english-vectors.html)  

I personally tried to use `crawl-300d-2M.vec` (embedding dimension=300) from fastText, and `glove.twitter.27B` (embedding dimension=200) and `glove.840B.300d` (embedding dimension=300) from GloVe.
