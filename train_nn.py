import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, log_loss
from utils import format_time, create_submission, average_predictions, \
        geom_average_predictions, read_yaml
from sklearn.model_selection import KFold, train_test_split
import time
from models import instantiate_model
import preprocessing
import running
import argparse

import sys

SEED = 2610
np.random.seed(SEED)
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

parser = argparse.ArgumentParser(description="Recurrent neural network for identifying and classifying toxic online comments")
parser.add_argument("embedding_file")
parser.add_argument("--model", default=None)

args = parser.parse_args()

print('Parsing parameters')
embedding_file = args.embedding_file

if args.model:
    model_name = args.model
    try:
        config = read_yaml('best_configs/{}.yaml'.format(model_name))
        print('Best parameters for {} loaded!'.format(model_name))
    except FileNotFoundError:
        print('Wrong model type!')
        sys.exit(1)
else:
    config = read_yaml('parameters.yaml')
    print('Custom parameters loaded!')

FILENAME = config.FILENAME
MODEL_TYPE = config.MODEL_TYPE
TRAINING_PARAMS = config.training_parameters
MODEL_PARAMS = config.model_parameters
PREPROCESSING_PARAMS = config.preprocessing_parameters

max_sequence_length = PREPROCESSING_PARAMS.max_sequence_length
max_nb_words = PREPROCESSING_PARAMS.max_nb_words

print('Loading and preprocessing data')
df, df_test = preprocessing.load_data(PREPROCESSING_PARAMS, embedding_file)

"""Classic NLP preprocessing for Keras"""
pp = preprocessing.Preprocessor('comment_text', PREPROCESSING_PARAMS)

df, df_test = pp.fill_null(df, df_test)
y_train = df[labels].values

pp.set_tokenizer(df, df_test, fit_on_train_only=False)
x_train, x_test = pp.tokenize_and_pad(df, df_test)
print('Shape of train tensor ', x_train.shape)
print('Shape of test tensor ', x_test.shape)

#Pre-trained embeddings
print('Creating embedding matrix...')
begin_matrix = time.time()
embedding_matrix, embedding_dimension = pp.make_words_vec(embedding_file)
end_matrix = format_time(time.time() - begin_matrix)
print('Matrix created - shape: {} - time: {}'.format(embedding_matrix.shape, end_matrix))
print(embedding_dimension)

"""Fit on 90% of the dataset"""
if config.run_90p:

    begin_cv = time.time()
    model = instantiate_model(MODEL_TYPE, MODEL_PARAMS, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix)
    print(model.summary())

    X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state = SEED)
    #filepath = 'temporary_cv_models/model-{}-autocheckpoint.h5'.format(FILENAME)
    filepath = 'model-{}-autocheckpoint.h5'.format(FILENAME)
    print('Fitting model on 90% of the data...')
    hist, stopped_epoch = running.fitting_model(model, X_train, Y_train, X_valid, Y_valid, TRAINING_PARAMS, filepath)
    end_cv = format_time(time.time() - begin_cv)

    print('Training complete - {} - time: {}'.format(FILENAME, end_cv))
    auc = hist['roc_auc_val'][stopped_epoch-TRAINING_PARAMS.patience]
    loss = hist['val_loss'][stopped_epoch-TRAINING_PARAMS.patience]
    print('AUC:', round(auc,5), ' / Loss:', round(loss,5))

    print('Generating predictions from the best model...')
    model.load_weights(filepath)
    preds_ = model.predict(x_test, batch_size = 2*TRAINING_PARAMS.batch_size, verbose = 1)
    create_submission(preds_, '{}_cv_{}-ave_{:0.5f}.csv'.format(MODEL_TYPE, FILENAME, auc))

"""Perform K fold CV"""
if config.run_kfold:
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)
    cv_scores = []
    cv_losses = []
    cv_predictions = []
    oof_predictions = np.zeros((len(x_train), len(labels)))

    begin_cv = time.time()
    print('Performing Kfold cross validation...')
    for i, (train, valid) in enumerate(kf.split(x_train)):
        begin_fold = time.time()
        model = instantiate_model(MODEL_TYPE, MODEL_PARAMS, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix)
        #filepath = 'temporary_cv_models/model-{}.fold-{:d}-autocheckpoint.h5'.format(FILENAME, i+1)
        filepath = 'model-{}.fold-{:d}-autocheckpoint.h5'.format(FILENAME, i+1)
        if i == 0:
            print(model.summary())
        print('Starting fold {:d}...'.format(i+1))
        hist, stopped_epoch = running.fitting_model(model, x_train[train], y_train[train], x_train[valid], y_train[valid], TRAINING_PARAMS, filepath)

        end_fold = format_time(time.time() - begin_fold)

        cv_scores.append(hist['roc_auc_val'][stopped_epoch-TRAINING_PARAMS.patience])
        cv_losses.append(hist['val_loss'][stopped_epoch-TRAINING_PARAMS.patience])
        print('--- Fold AUC {:.5f} - Stopped epoch: {:d} - time: {}'.format(cv_scores[i], stopped_epoch-TRAINING_PARAMS['patience']+1, end_fold))
        print('--- Fold loss {:.5f}'.format(cv_losses[i]))
        #load the best model to generate OOF predictions and test predictions
        model.load_weights(filepath)
        print('Saving OOF predictions...')
        oof_predictions[valid, :] = model.predict(x_train[valid], batch_size = 2*TRAINING_PARAMS.batch_size, verbose = 1)
        #pd.DataFrame(oof_predictions).to_csv('temporary_oof_preds/temp_oof_preds_'+FILENAME+'.csv',index=False)
        pd.DataFrame(oof_predictions).to_csv('temp_oof_preds_'+FILENAME+'.csv',index=False)

        print('Generating test predictions from the best model...')
        preds_ = model.predict(x_test, batch_size = 2*TRAINING_PARAMS.batch_size, verbose = 1)
        cv_predictions.append(preds_)
        #pd.DataFrame(preds_).to_csv('temporary_cv_preds/preds-{}.fold-{:d}.csv'.format(FILENAME,i+1), index = False)
        pd.DataFrame(preds_).to_csv('preds-{}.fold-{:d}.csv'.format(FILENAME,i+1), index = False)
        print('\n')
        del model

    end_cv = format_time(time.time() - begin_cv)
    #pd.DataFrame(oof_predictions).to_csv('oof_predictions/oof_pred_'+FILENAME+'.csv', index = False)
    pd.DataFrame(oof_predictions).to_csv('oof_pred_'+FILENAME+'.csv', index = False)
    oof_auc = roc_auc_score(y_train, oof_predictions)

    print('--- Results : {}\nAverage AUC {:.5f} +/- {:.4f} -- time: {}'.format(FILENAME, np.mean(cv_scores), np.std(cv_scores), end_cv))
    print('Average loss {:.5f} +/- {:.4f}'.format(np.mean(cv_losses), np.std(cv_losses)))
    print('Out of fold AUC {:.5f}'.format(oof_auc))

    """Compute n-folds average test predictions"""
    print('Saving predictions...')
    preds = average_predictions(cv_predictions, config.n_folds)
    create_submission(preds, '{}_cv_{}.csv'.format(MODEL_TYPE, FILENAME))

    #"""Geometric mean"""
    #print('Geometric')
    #preds2 = geom_average_predictions(cv_predictions, config.n_folds)
    #create_submission(preds2, '{}_cvgeom_{}.csv'.format(MODEL_TYPE, FILENAME))
