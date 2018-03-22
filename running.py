from utils import RocAucEvaluation,DisplayLR
from keras.callbacks import EarlyStopping, ModelCheckpoint
def fitting_model(model, X_train, Y_train, X_valid, Y_valid, TRAINING_PARAMS, filepath):
    lrdisplay = DisplayLR()
    ra_val = RocAucEvaluation(validation_data=(X_valid, Y_valid))
    checkpoint = ModelCheckpoint(filepath, monitor=TRAINING_PARAMS['monitored_value'], verbose=1, save_best_only=True, save_weights_only = True)
    es = EarlyStopping(monitor=TRAINING_PARAMS['monitored_value'], min_delta = TRAINING_PARAMS['min_delta'],
                                patience=TRAINING_PARAMS['patience'], mode='auto', verbose=1)

    hist = model.fit(X_train, Y_train, epochs = TRAINING_PARAMS['nb_epochs'], batch_size= TRAINING_PARAMS['batch_size'],
                    callbacks = [ra_val, checkpoint, es], validation_data = (X_valid, Y_valid))
    return hist.history, es.stopped_epoch
