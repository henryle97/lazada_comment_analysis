import keras_self_attention

from model_util import model
from model_util.model import SARNN_Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pyvi import ViTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
from extract_feature import make_embedding, texts_to_sequences, tokenize
from data.utils import read_file
import os
from keras.utils import to_categorical
from keras.models import load_model
from extract_feature import processing_data
import model_util

class SENTIMENT_CLASSIFY:
    def __init__(self):
        self.model_path = "models/SARNN-version/models.hdf5"
        self.batch_size = 50
        self.epochs = 100

    def clean_data(self, csv_path, csv_result_path):
        processing_data(csv_path, csv_result_path)
        return csv_result_path

    def create_feature(self, data_path, embedding_path, max_features):
        '''

        :param data_path: file txt/csv chứa comment và label
        :return: sequence
        '''
        train_data = read_file(data_path)
        train_tokenized_texts = tokenize(train_data['comment'])

        labels = train_data['stars'].tolist()
        labels = to_categorical(labels)  #return encode cho 0->numclass
        labels = labels[:,labels.any(0)] # bỏ đi các cột ko có giá trị khác không nào

        test_data = read_file("data/train_500.csv")
        test_tokenized_texts = tokenize(test_data['comment'])
        labels_test = test_data['stars'].tolist()
        labels_test = to_categorical(labels_test)  # return encode cho 0->numclass
        labels_test = labels_test[:, labels_test.any(0)]  # bỏ đi các cột ko có giá trị khác không nào

        train_tokenized_texts, val_tokenized_texts, labels_train, labels_val = train_test_split(
            train_tokenized_texts, labels, test_size=0.1, random_state=1997
        )

        # Cần tạo embedding size cho cả train và test
        embed_size, word_map, embedding_mat = make_embedding(
            list(train_tokenized_texts) + list(val_tokenized_texts) + list(test_tokenized_texts),
            embedding_path,
            max_features
        )
        texts_id_train = texts_to_sequences(train_tokenized_texts, word_map)
        texts_id_val = texts_to_sequences(val_tokenized_texts, word_map)
        texts_id_test = texts_to_sequences(test_tokenized_texts, word_map)
        return embed_size, word_map, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val, texts_id_test, labels_test

    def training_ML(self):
        X_train = np.load(self.features_path, allow_pickle=True)
        y_train = np.load(self.labels_path)
        model = LogisticRegression()

        # CROSS VALIDATION
        score = cross_val_score(model, X_train, y_train, cv=8)
        print(score)

        # TRAINING
        model.fit(X_train, y_train)
        pickle.dump(model, open(self.model_path, 'wb'))

    def training_sarnn(self, embed_size, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val,
                       trainable=False, use_additive_emb=True):

        model_name = "SARNN"
        print('Number of train data: {}'.format(len(labels_train.tolist() + labels_val.tolist())))

        # texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
        #     texts_id, labels, test_size=0.05)

        model_path = './models/{}-version'.format(model_name)

        try:
            os.mkdir('./models')
        except:
            print('Folder already created')
        try:
            os.mkdir(model_path)
        except:
            print('Folder already created')

        checkpoint = ModelCheckpoint(
            filepath='{}/models.hdf5'.format(model_path),
            monitor='val_f1', verbose=1,
            mode='max',
            save_best_only=True
        )
        early = EarlyStopping(monitor='val_f1', mode='max', patience=5)
        callbacks_list = [checkpoint, early]


        model = SARNN_Keras(
            embeddingMatrix=embedding_mat,
            embed_size=embed_size,
            max_features=embedding_mat.shape[0],
            trainable=trainable,
            use_additive_emb=use_additive_emb
        )

        model.fit(
            texts_id_train, labels_train,
            validation_data=(texts_id_val, labels_val),
            callbacks=callbacks_list,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def predict(self, X_test, y_test):
        # load model_util
        model = load_model(self.model_path, custom_objects={'SeqSelfAttention' : keras_self_attention.SeqSelfAttention,
                                                            'SeqWeightedAttention':keras_self_attention.SeqWeightedAttention,'f1': model_util.utils.f1})

        # predict
        result = model.predict(X_test)
        print(result)

if __name__ == "__main__":
    sa = SENTIMENT_CLASSIFY()
    embed_size, word_map, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val, texts_id_test, labels_test = sa.create_feature(
        data_path="data/data_hoang_processed.csv", embedding_path="data/baomoi.model.bin", max_features=120000
    )
    print(len(list(texts_id_train)))
    print(texts_id_train.shape)
    print(labels_train.shape)
    sa.training_sarnn( embed_size, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val,
                       trainable=True, use_additive_emb=False)
    # sa.predict(texts_id_test, labels_test)