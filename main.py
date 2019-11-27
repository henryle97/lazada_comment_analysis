from model import model
from model.model import SARNN_Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pyvi import ViTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import pickle
from extract_feature import make_embedding, texts_to_sequences, tokenize
from data.utils import read_file
import os
from tensorflow.keras.utils import to_categorical

class SENTIMENT_CLASSIFY:
    def __init__(self):
        self.features_path = "data/features_tfidf.npy"
        self.labels_path = "data/labels_tfidf.npy"
        self.model_path = "model/model.hdf5"

    def create_feature(self, data_path, embedding_path, max_features):
        '''

        :param data_path: file txt/csv chứa comment và label
        :return: sequence
        '''
        train_data = read_file(data_path)
        train_tokenized_texts = tokenize(train_data['comment'])
        labels = train_data['stars'].tolist()
        print(labels[:10])
        labels = to_categorical(labels)
        labels = labels[:,labels.any(0)]
        print(labels.shape)

        train_tokenized_texts, val_tokenized_texts, labels_train, labels_val = train_test_split(
            train_tokenized_texts, labels, test_size=0.05, random_state=1997
        )

        # Cần tạo embedding size cho cả train và test
        embed_size, word_map, embedding_mat = make_embedding(
            list(train_tokenized_texts) + list(val_tokenized_texts),
            embedding_path,
            max_features
        )
        texts_id_train = texts_to_sequences(train_tokenized_texts, word_map)
        texts_id_val = texts_to_sequences(val_tokenized_texts, word_map)
        return embed_size, word_map, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val

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
        batch_size = 16
        epochs = 100

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
            epochs=epochs,
            batch_size=batch_size
        )

    def predict(self, X_test, y_test):
        # load model
        model = pickle.load(open(self.model_path, 'rb'))

        # predict
        result = model.score(X_test, y_test)


if __name__ == "__main__":
    sa = SENTIMENT_CLASSIFY()
    embed_size, word_map, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val = sa.create_feature(
        data_path="data/data_hoang_processed.csv", embedding_path="data/baomoi.model.bin", max_features=120000
    )
    print(len(list(texts_id_train)))
    print(texts_id_train.shape)
    print(labels_train.shape)
    sa.training_sarnn( embed_size, embedding_mat, texts_id_train, texts_id_val, labels_train, labels_val,
                       trainable=True, use_additive_emb=False)