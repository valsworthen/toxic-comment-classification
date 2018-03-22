# How to run  

Make sure you have Tensorflow **1.5** installed on GPU.  

The script saves a lot of files (OOF, predictions, models) in the following folders:  
- `temporary_cv_models/`  
- `temporary_cv_preds/`  
- `temporary_oof_preds/`  

Input data should be inside `input/` folder.  

**Make sure to replace the** `embedding` **dictionnary in**`train_nn.py` **with your paths to pre-trained embeddings** (see below if you're not familiar with pre-trained embeddings)  

The *main* file is `train_nn.py`.  
When run, this file asks for a **model type**. This corresponds to models implemented in `models.py`. At the moment, models implemented are:  
- bibigru  
- pooled_gru  
- ngram_cnn  
- cnn_gru  
- gru_cnn  

Feel free to complete `models.py` with your own architecture!  

##Models  

TBD: list the different models I used during the competitions

## Embeddings  

Word embeddings are used in NLP to compute *vector representation* of words that can be fed to machine learning models. Such representations can be extracted from statistics of the corpus but it is now common to *learn* them via a "fake task". You can check [this great explanation of Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/), one of the most famous algorithm to learn word embeddings.  

However, learning precise representations require **a lot** of training data and computing power. That's why it is common to download and use **pre-trained embeddings** such as:
- [GloVe](https://nlp.stanford.edu/projects/glove/)  
- [FastText](https://fasttext.cc/docs/en/english-vectors.html)  

I personally tried to use `crawl-300d-2M.vec` (embedding dimension=300) from fastText, and `glove.twitter.27B` (embedding dimension=200) and `glove.840B.300d` (embedding dimension=300) from GloVe.
