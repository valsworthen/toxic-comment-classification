"""Utilities"""
import pandas as pd
import numpy as np
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

"""GloVe/fastText embeddings"""
import numpy as np

from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import time
class RocAucEvaluation(Callback):
    """This callback computes AUC on the validation data which allows us to monitor training"""
    """We should modify this callback to exclude checkpoint and checkpoint.
    These should be executed with Keras built-in callbacks (see https://www.kaggle.com/tilii7/keras-averaging-runs-gini-early-stopping)"""
    def __init__(self, filepath = None, validation_data=()):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.best = 0
        self.X_val, self.y_val = validation_data
        self.y_pred = np.zeros(self.y_val.shape)

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0)
        begin_pred = time.time()
        current = roc_auc_score(self.y_val, y_pred)
        end_pred = format_time(time.time() - begin_pred)
        logs['roc_auc_val'] = current

        if current > self.best: #save model
            self.best = current
            self.y_pred = y_pred
            if self.filepath:
                self.model.save(self.filepath, overwrite = True)
        print("--- AUC - epoch: {:d} - score: {:.5f} - time: {}".format(epoch+1, current, end_pred))
"""
use checkpoint AFTER early stopping
EarlyStopping(monitor='roc_auc_val', patience=patience, mode='max', verbose=1)
ModelCheckpoint(filepath, monitor='roc_auc_val', mode='max', save_best_only=True, verbose=1)
"""
import keras.backend as K
class DisplayLR(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print('\nLearning rate for next epoch: ',K.eval(lr_with_decay))
