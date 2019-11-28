
import keras_self_attention
from model_util.model import SARNN_Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.model_selection import  train_test_split
import pickle
from extract_feature import make_embedding, texts_to_sequences, tokenize
from data.utils import read_file
import os
from keras.utils import to_categorical
from keras.models import load_model
from extract_feature import processing_data
import model_util
from sklearn.metrics import classification_report

class SENTIMENT_CLASSIFY:
    def __init__(self):
        self.model_path = "models.hdf5"
        self.batch_size = 128
        self.epochs = 100
        self.data_train_path = "data/data_all_processed_sorted.csv"
        self.data_test_path = "data/test_287.csv"
        self.embedding_path = "embedding/baomoi.model.bin"
        self.word_map_path = "embedding/wordmap.pkl"
        self.embedding_matrix_path = "embedding/embeding_matrix.pkl"
        self.max_features = 120000
        self.model_name = "SARNN"

    def clean_data(self, csv_path, is_data_train):
        result_path = "data/"+os.path.basename(csv_path).split(".")[0] + "_processed.csv"
        processing_data(csv_path, result_path, is_data_train)

    def create_embedding(self):
        print("Creating embeding...")
        if not os.path.exists("embedding"):
            os.mkdir("embedding")
        train_data = read_file(self.data_train_path)
        train_tokenized_texts = tokenize(train_data['comment'])
        test_data = read_file(self.data_test_path)
        test_tokenized_texts = tokenize(test_data['comment'])

        embed_size, word_map, embedding_mat = make_embedding(
            list(train_tokenized_texts) + list(test_tokenized_texts),
            self.embedding_path,
            self.max_features
        )

        with open(self.word_map_path, 'wb') as f:
            pickle.dump(dict(word_map), f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.embedding_matrix_path, 'wb') as f:
            pickle.dump(embedding_mat, f, protocol=pickle.HIGHEST_PROTOCOL)

    def create_feature(self, data_path):
        '''

        :param data_path: file txt/csv chứa comment và label
        :return: sequence
        '''
        print("Creating feature ...")
        # load word_map
        if not os.path.exists(self.word_map_path):
            self.create_embedding()
        word_map = pickle.load(open(self.word_map_path, 'rb'))

        data = read_file(data_path)
        tokenized_texts = tokenize(data['comment'])
        texts_id = texts_to_sequences(tokenized_texts, word_map, max_len=80)

        labels = data['stars'].tolist()
        labels = to_categorical(labels)  #return encode cho 0->numclass
        labels = labels[:, labels.any(0)] # bỏ đi các cột ko có giá trị khác không nào

        return texts_id, labels

    def training_sarnn(self, trainable_embedding=False, use_additive_emb=False, re_create_embedding=True):
        if re_create_embedding:
            self.create_embedding()
        # load embedding
        if not os.path.exists(self.embedding_matrix_path):
            self.create_embedding()
        embedding_mat = pickle.load(open(self.embedding_matrix_path, 'rb'))
        embed_size = embedding_mat.shape[1]
        print(f"embed_size = {embed_size}")
        # load data train
        texts_id, labels = self.create_feature(data_path=self.data_train_path)

        texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
            texts_id, labels, test_size=0.1, random_state=1997, shuffle=True)
        print('Number of train data: {}'.format(len(list(texts_id))))

        model_path = './models/{}-version'.format(self.model_name)
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
        early = EarlyStopping(monitor='val_f1', mode='max', patience=10)
        callbacks_list = [checkpoint, early]


        model = SARNN_Keras(
            embeddingMatrix=embedding_mat,
            embed_size=embed_size,
            max_features=embedding_mat.shape[0],
            trainable_embedding=trainable_embedding,
            use_additive_emb=use_additive_emb
        )

        model.fit(
            texts_id_train, labels_train,
            validation_data=(texts_id_val, labels_val),
            callbacks=callbacks_list,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

    def predict(self):
        # load model_util
        X_test, y_test = self.create_feature(self.data_test_path)
        model = load_model(self.model_path, custom_objects={'SeqSelfAttention' : keras_self_attention.SeqSelfAttention,
                                                            'SeqWeightedAttention':keras_self_attention.SeqWeightedAttention,'f1': model_util.utils.f1})

        # predict
        result = model.predict(X_test)
        y_pred = np.argmax(result, axis=1)
        y_test = np.argmax(y_test, axis=1)
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    sa = SENTIMENT_CLASSIFY()
    sa.clean_data("/home/hisiter/IT/4_year_1/Intro_ML/lazada_comment_analysis/data/data_all_processed_sorted.csv", is_data_train=False)
    # sa.training_sarnn(trainable_embedding=True)
    # sa.predict()

'''
Result training 
- (1) : train_able = False 
+ f1 = 0.87 , val_f1 : 0.57864
+ test accuracy = 0.68


- (2): train_able = True, patient = 8
'''