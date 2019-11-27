from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    CuDNNGRU, GRU, LSTM, Bidirectional, CuDNNLSTM, \
    GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, \
    Lambda, Concatenate, TimeDistributed

from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras_layer_normalization import LayerNormalization
from model_util.net_component import AdditiveLayer, AttLayer
from model_util.utils import f1
from sklearn.svm import SVC

def SVC_Classification():
    model = SVC(kernel='rbf', C=1, gamma=1)
    return model

def SARNN_Keras(embeddingMatrix=None, embed_size=400, max_features=20000, maxlen=100,use_fasttext=False, trainable=True, use_additive_emb=False, rnn_type=LSTM, use_gpu=False):
    '''
    SARNN - Self-Attention RNN
    :param embeddingMatrix:
    :param embed_size:
    :param max_features:
    :param maxlen:
    :param use_fasttext:
    :param trainable:
    :param use_additive_emb:
    :param rnn_type:
    :param use_gpu:
    :return:
    '''
    if use_gpu:
        rnn_type = CuDNNLSTM

    input = Input(shape=(maxlen, ))
    x = Embedding(input_dim=max_features, output_dim=embed_size, weights=[embeddingMatrix], trainable=trainable)(input)

    if use_additive_emb:
        x = AdditiveLayer()(x)
        x = Dropout(0.5)(x)

    x = Bidirectional(rnn_type(128, return_sequences=True))(x)
    x = SeqSelfAttention(
        #attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL
        attention_regularizer_weight=1e-4
    )(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(rnn_type(128, return_sequences=True))(x)
    x = SeqWeightedAttention()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(5, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', f1])
    return model
