# PREPROCESSING PART
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
import pandas as pd

class Preprocessor():
    def __init__(self, TEXT_COLUMN, PREPROCESSING_PARAMS):
        self.TEXT_COLUMN = TEXT_COLUMN
        self.max_nb_words = PREPROCESSING_PARAMS.max_nb_words
        self.max_sequence_length = PREPROCESSING_PARAMS.max_sequence_length

    def fill_null(self, train, test, pattern='no comment'):
        train[self.TEXT_COLUMN] = train[self.TEXT_COLUMN].fillna(pattern)
        test[self.TEXT_COLUMN] = test[self.TEXT_COLUMN].fillna(pattern)
        return train, test

    def set_tokenizer(self, train, test, fit_on_train_only=False):
        tokenizer = Tokenizer(num_words=self.max_nb_words, char_level=False, lower = True)
        if fit_on_train_only:
            tokenizer.fit_on_texts(train[self.TEXT_COLUMN].tolist())
        else:
            tokenizer.fit_on_texts(train[self.TEXT_COLUMN].tolist() + test[self.TEXT_COLUMN].tolist())
        self.tokenizer = tokenizer

    def tokenize_and_pad(self, train, test):
        sequences_train = self.tokenizer.texts_to_sequences(train[self.TEXT_COLUMN])
        sequences_test = self.tokenizer.texts_to_sequences(test[self.TEXT_COLUMN])

        x_train = pad_sequences(sequences_train, maxlen=self.max_sequence_length)
        x_test = pad_sequences(sequences_test, maxlen=self.max_sequence_length)
        return x_train, x_test

    def make_words_vec(self, EMBEDDING_FILE):
        def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
        """compute word vectors for our corpus"""
        embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(EMBEDDING_FILE)))

        embedding_dimension = len(next (iter (embedding_index.values())))

        word_index = self.tokenizer.word_index
        nb_words = min(self.max_nb_words, len(word_index))
        embedding_matrix = np.zeros((nb_words, embedding_dimension))
        for word, i in word_index.items():
            if i >= self.max_nb_words: continue
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix, embedding_dimension

def load_data(PREPROCESSING_PARAMS, embedding_file):
    #data preprocessed by Zafar
    #https://www.kaggle.com/fizzbuzz/cleaned-toxic-comments
    if PREPROCESSING_PARAMS.use_preprocessed_data:
        df = pd.read_csv('input/train_preprocessed.csv')
        df_test = pd.read_csv('input/test_preprocessed.csv')
    #data preprocessed for Twitter embeddings
    if 'twitter' in embedding_file:
        print('Loading twitter dataframe')
        df = pd.read_csv('input/train_twitter.csv')
        df_test = pd.read_csv('input/test_twitter.csv')
    else:
        df = pd.read_csv('input/train.csv')
        df_test = pd.read_csv('input/test.csv')

    return df,df_test

import re
def preprocess(train, test):
    repl = {
        "&lt;3": " good ",
        ":d": " good ",
        ":dd": " good ",
        ":p": " good ",
        "8)": " good ",
        ":-)": " good ",
        ":)": " good ",
        ";)": " good ",
        "(-:": " good ",
        "(:": " good ",
        "yay!": " good ",
        "yay": " good ",
        "yaay": " good ",
        "yaaay": " good ",
        "yaaaay": " good ",
        "yaaaaay": " good ",
        ":/": " bad ",
        ":&gt;": " sad ",
        ":')": " sad ",
        ":-(": " bad ",
        ":(": " bad ",
        ":s": " bad ",
        ":-s": " bad ",
        "&lt;3": " heart ",
        ":d": " smile ",
        ":p": " smile ",
        ":dd": " smile ",
        "8)": " smile ",
        ":-)": " smile ",
        ":)": " smile ",
        ";)": " smile ",
        "(-:": " smile ",
        "(:": " smile ",
        ":/": " worry ",
        ":&gt;": " angry ",
        ":')": " sad ",
        ":-(": " sad ",
        ":(": " sad ",
        ":s": " sad ",
        ":-s": " sad ",
        r"\br\b": "are",
        r"\bu\b": "you",
        r"\bhaha\b": "ha",
        r"\bhahaha\b": "ha",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bhasn't\b": "has not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bcan't\b": "can not",
        r"\bcannot\b": "can not",
        r"\bi'm\b": "i am",
        "m": "am",
        "r": "are",
        "u": "you",
        "haha": "ha",
        "hahaha": "ha",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "can not",
        "cannot": "can not",
        "i'm": "i am",
        "m": "am",
        "i'll" : "i will",
        "its" : "it is",
        "it's" : "it is",
        "'s" : " is",
        "that's" : "that is",
        "weren't" : "were not",
    }

    keys = [i for i in repl.keys()]

    new_train_data = []
    new_test_data = []
    ltr = train["comment_text"].tolist()
    lte = test["comment_text"].tolist()
    for i in ltr:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + " "
        new_train_data.append(xx)
    for i in lte:
        arr = str(i).split()
        xx = ""
        for j in arr:
            j = str(j).lower()
            if j[:4] == 'http' or j[:3] == 'www':
                continue
            if j in keys:
                # print("inn")
                j = repl[j]
            xx += j + " "
        new_test_data.append(xx)
    train["new_comment_text"] = new_train_data
    test["new_comment_text"] = new_test_data
    print("crap removed")
    trate = train["new_comment_text"].tolist()
    tete = test["new_comment_text"].tolist()
    for i, c in enumerate(trate):
        trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
    for i, c in enumerate(tete):
        tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
    train["comment_text"] = trate
    test["comment_text"] = tete
    print('only alphabets')

    return train, test
