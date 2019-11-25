import os
import glob
import numpy as np
from pyvi import ViTokenizer  # co the thay bang # https://github.com/trungtv/vi_spacy
import string
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict

label_mapping = {"negative":0, "neutral":1, "positive":2}
class DataLoader:
    def load_data(self, data_dir):
        file_paths = [path for path in glob.glob(data_dir + "/*.txt")]
        data = []
        labels = []

        for file_path in file_paths:
            label_file = os.path.basename(file_path).split('.txt')[0].split("_")[-1]
            data_lines = list(filter(lambda x: x != '', open(file_path).read().split('\n')))
            label = np.full((len(data_lines),), label_mapping[label_file])
            data = data + data_lines
            labels = labels + label.tolist()
        return data, labels

    def tokenize(self, texts):
        nlp = ViTokenizer
        docs = []
        for text in texts:
            text = text.replace()
            tokens = [self._postprocess_token(token) for token in nlp.tokenize(text.lower()).split(' ')]
            docs.append(tokens)

        return docs
    def _postprocess_token(self, token):
        if token in string.punctuation:
            return "<punct>"
        elif token.isdigit():
            return "<number>"
        else:
            return token

    def make_embedding(self, texts, embedding_path, max_features):
        '''

        :param texts:
        :param embedding_path:
        :param max_features:
        :return:
        '''
        # embedding_path = os.path.abspath(embedding_path)

        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        if embedding_path.endswith('.vec'):
            embedding_index = dict(get_coefs(*o.strip().split(" "))
                                   for o in open(embedding_path))
            mean_embedding = np.mean(np.array(list(embedding_index.values())))
        elif embedding_path.endswith('bin'):
            embedding_index = KeyedVectors.load_word2vec_format(
                embedding_path, binary=True)
            mean_embedding = np.mean(embedding_index.vectors, axis=0)

        embed_size = mean_embedding.shape[0]
        word_index = sorted(list({word.lower() for sentence in texts for word in sentence}))
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.zeros((nb_words+1, embed_size))
        i = 1
        word_map = defaultdict(lambda: nb_words)
        for word in word_index:
            if i >= max_features:
                continue
            if word in embedding_index:
                embedding_matrix[i] = embedding_index[word]
            else:
                embedding_matrix[i] = mean_embedding
            word_map[word] = i
            i += 1
        embedding_matrix[-1] = mean_embedding

        return embed_size, word_map, embedding_matrix

    def texts_to_sequences(self, texts, word_map, max_len=DEFAULT_MAX_LENGTH):
        '''
        Chuyển text = list(word) thành list(id) tương ứng trong embedding
        :param texts:
        :param word_map:
        :param max_len:
        :return:
        '''
        texts_id = []
        for sentence in texts:
            sentence = [word_map[word.lower()] for word in sentence][:max_len]
            padded_sentence = np.pad(sentence, (0, max(0, max_len - len(sentence))), 'constant', constant_values=0)
            texts_id.append(padded_sentence)
        return np.array(texts_id)


if __name__ == "__main__":
    dataloader = DataLoader()
    data, label = dataloader.load_data("/home/hisiter/IT/4_year_1/Intro_ML/sentiment_classification/SA2016-training_data")
    docs = dataloader.tokenize(data)
    print(docs[0:10])
    print("@@@" in string.punctuation)


# TODO

# https://github.com/nhatthanh123bk/sentiment-analysis/blob/master/utils.py
# Replace special characters: @@, ....