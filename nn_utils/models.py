from keras import optimizers
from keras.models import Model, Input
from keras.layers import Embedding, Dense
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import SpatialDropout1D, PReLU, BatchNormalization, Dropout
from keras.layers import Conv1D, Conv2D
from nn_utils.attlayer import AttentionWeightedAverage

def instantiate_model(model_type, model_params, *args):
    """
    From the model type and parameters, this function instantiates a Keras Model
    It can be used to create any type of model, and not only RNN (see build_test)
    """
    m = ModelBuilder(model_params)
    models = {'bibigru':'build_bibigru',
                'gru_cnn': 'build_gru_cnn',
                'pooled_gru':'build_pooled_gru',
                'cnn_gru':'build_cnn_gru',
                'ngram_cnn':'build_ngram_cnn',
                'test':'build_test'}

    if model_type in models:
        builder_name = models[model_type]
        builder = getattr(m, builder_name)
        return builder(*args)#max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix)

    else:
        raise Exception("Model %s not implemented" % model_type)

class ModelBuilder():
    def __init__(self,model_params):
        self.dr = model_params['dr']
        self.gru_units = model_params['gru_units']
        self.ngram_range = (model_params['ngram_range'][0], model_params['ngram_range'][1]+1)
        self.num_filters = model_params['num_filters']
        self.use_attention = bool(model_params['use_attention'])
        self.use_maxpool = bool(model_params['use_maxpool'])
        self.use_avgpool = bool(model_params['use_avgpool'])
        self.use_dense = bool(model_params['use_dense'])
        self.dense_size = model_params['dense_size']

        self.optimizer = model_params['optimizer']
        self.lr = model_params['lr']
        self.decay = model_params['decay']

    def build_optimizer(self):
        #Nadam default: lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004
        #Adam default: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
        if self.optimizer == 'adam':
            return optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=self.decay, amsgrad=False)
        elif self.optimizer == 'nadam':
            return optimizers.Nadam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=self.decay)

    def build_conv_blocks(self,x):
        #for each k-gram we create a conv block. All the blocks will be concatenated
        conv_blocks = []
        for ks in range(*self.ngram_range):
            kgram_layers = []
            kgram = Conv1D(filters = self.num_filters,kernel_size = ks,padding='same',activation='relu',strides=1, name = str(ks)+'gram_conv')(x)

            if self.use_attention:
                katt = AttentionWeightedAverage(name=str(ks)+'gram_att')(kgram)
                kgram_layers.append(katt)
            if self.use_maxpool:
                kmax = GlobalMaxPooling1D(name=str(ks)+'gram_maxpool')(kgram)
                kgram_layers.append(kmax)
            if self.use_avgpool:
                kavg = GlobalAveragePooling1D(name=str(ks)+'gram_avgpool')(kgram)
                kgram_layers.append(kavg)
            if len(kgram_layers) > 1:
                kgram = concatenate(kgram_layers, name = str(ks)+'gram_concat')
            else: kgram = kgram_layers[0]
            conv_blocks.append(kgram)
        return conv_blocks

    def build_bibigru(self, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix):
        inp = Input(shape=(max_sequence_length,))
        x = Embedding(max_nb_words, embedding_dimension, weights=[embedding_matrix], trainable = False)(inp)
        x = SpatialDropout1D(self.dr)(x)

        gru_seq_1 = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, name = 'bigru1'))(x)
        gru_seq_2, gru_state, _ = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, return_state = True, name = 'bigru2'))(gru_seq_1)
        x = concatenate([gru_seq_2, gru_seq_1, x])
        x = AttentionWeightedAverage(name='attlayer')(x)

        if self.use_dense:
            x = BatchNormalization()(x)
            x = Dense(self.dense_size, activation = 'relu')(x)
            x = BatchNormalization()(x)

        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_gru_cnn(self, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix):
        inp = Input(shape=(max_sequence_length,))
        x = Embedding(max_nb_words, embedding_dimension, weights=[embedding_matrix], trainable = False)(inp)
        x = SpatialDropout1D(self.dr)(x)
        gru_seq_1 = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, name = 'bigru1'))(x)
        gru_seq_2, gru_state, _ = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, return_state = True, name = 'bigru2'))(gru_seq_1)
        conc = concatenate([gru_seq_2, gru_seq_1, x])

        #for each k-gram we create a conv block. All the blocks will be concatenated
        conv_blocks = self.build_conv_blocks(conc)
        if len(conv_blocks) > 1:
            x = concatenate(conv_blocks)
        else: x = conv_blocks[0]

        if self.use_dense:
            x = BatchNormalization()(x)
            x = Dense(self.dense_size, activation = 'relu')(x)
            x = BatchNormalization()(x)

        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_pooled_gru(self, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix):
        inp = Input(shape=(max_sequence_length,))
        x = Embedding(max_nb_words, embedding_dimension, weights=[embedding_matrix], trainable = False)(inp)
        x = SpatialDropout1D(self.dr)(x)
        gru_seq, gru_state, _ = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, return_state = True))(x)
        maxpool = GlobalMaxPooling1D()(gru_seq)
        avgpool = GlobalAveragePooling1D()(gru_seq)
        x = concatenate([avgpool, maxpool, gru_state])

        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_cnn_gru(self, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix):
        inp = Input(shape=(max_sequence_length,))
        x = Embedding(max_nb_words, embedding_dimension, weights=[embedding_matrix], trainable = False)(inp)
        x = SpatialDropout1D(self.dr)(x)

        seq = Bidirectional(CuDNNGRU(self.gru_units, return_sequences=True, name='bigru'))(x)
        seq = [AttentionWeightedAverage(name='gru_attlayer')(seq)]
        #for each k-gram we create a conv block. All the blocks will be concatenated
        conv_blocks = self.build_conv_blocks(x)
        x = concatenate(seq+conv_blocks)

        if self.use_dense:
            x = BatchNormalization()(x)
            x = Dense(self.dense_size, activation = 'relu')(x)
            x = BatchNormalization()(x)

        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_ngram_cnn(self, max_sequence_length, max_nb_words, embedding_dimension, embedding_matrix):
        inp = Input(shape=(max_sequence_length,))
        x = Embedding(max_nb_words, embedding_dimension, weights=[embedding_matrix], trainable = False)(inp)
        x = SpatialDropout1D(self.dr)(x)

        #for each k-gram we create a conv block. All the blocks will be concatenated
        conv_blocks = self.build_conv_blocks(x)
        if len(conv_blocks) > 1:
            x = concatenate(conv_blocks)
        else: x = conv_blocks[0]

        if self.use_dense:
            x = BatchNormalization()(x)
            x = Dense(self.dense_size, activation = 'relu')(x)
            x = BatchNormalization()(x)

        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def build_test(self):
        inp = Input(shape=(100,100,3))
        x = Conv2D(512,(3,3))(inp)
        x = Flatten()(x)
        outp = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=outp)

        optimizer = self.build_optimizer()
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
