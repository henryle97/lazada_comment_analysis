from model import model
from model.model import SARNN_Keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pyvi import ViTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

class SENTIMENT_CLASSIFY:
    def __init__(self):
        self.features_path = "data/features_tfidf.npy"
        self.labels_path = "data/labels_tfidf.npy"
        self.model_path = "model/model.hdf5"

    def create_feature(self, data_path):
        '''

        :param data_path: file txt/csv chứa comment và label
        :return:
        '''

    def training(self):
        X_train = np.load(self.features_path, allow_pickle=True)
        y_train = np.load(self.labels_path)
        model = LogisticRegression()

        # CROSS VALIDATION
        score = cross_val_score(model, X_train, y_train, cv=8)
        print(score)

        # TRAINING
        model.fit(X_train, y_train)
        pickle.dump(model, open(self.model_path, 'wb'))


    def predict(self, X_test, y_test):
        # load model
        model = pickle.load(open(self.model_path, 'rb'))

        # predict
        result = model.score(X_test, y_test)


if __name__ == "__main__":
    sa = SENTIMENT_CLASSIFY()
    sa.training()