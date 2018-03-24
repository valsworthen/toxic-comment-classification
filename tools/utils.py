"""Utilities"""
import pandas as pd
import numpy as np
from attrdict import AttrDict
import yaml

def average_predictions(cv_predictions, n_splits, num_samples = 153164, num_labels = 6):
    """Average k-fold predictions stored in a dict"""
    preds = np.zeros((num_samples, num_labels))
    for preds_i in cv_predictions:
        preds += preds_i
    preds /= n_splits
    return preds

def geom_average_predictions(cv_predictions, n_splits, num_samples = 153164, num_labels = 6):
    """Average k-fold predictions stored in a dict"""
    preds = np.ones((num_samples, num_labels))
    for preds_i in cv_predictions:
        preds *= preds_i
    preds = preds **(1/n_splits)
    return preds

def create_submission(preds, filename):
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    subm = pd.read_csv('input/sample_submission.csv')
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(preds, columns = labels)], axis=1)
    submission.to_csv(filename, index=False)

def format_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "{:.0f}h {:.0f}min {:.0f}s".format(h, m, s)

def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)
