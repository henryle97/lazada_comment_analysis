from collections import defaultdict
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import time
import re
import unicodedata
from pyvi import ViTokenizer
import json

DEFAULT_MAX_FEATURES = 12000
DEFAULT_MAX_LENGTH = 100


# PREPROCESSING
def preprocessing_comment(text):
    text = text.lower()
    text = unicodedata.normalize('NFC', text)  #   Unicode normalization form
    text = re.sub("[^a-zA-ZàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]", " ", text)

    with open('data/replace_dict.json', 'r') as JSON:
        replace_dict = json.load(JSON)
    text = " ".join(replace_dict[token] if token in replace_dict else token for token in text.split(" "))
    text = ViTokenizer.tokenize(text)
    return text


def processing_data(csv_path, csv_result_path="train_processed.csv", data_train=True):
    t1 = time.time()
    print("Start processing data ...")
    df = pd.read_csv(csv_path, sep=",", usecols=range(2), names=["stars", "comment"])
    if data_train == True:
        df_drop = df.drop_duplicates(keep='first', inplace=False)
    df_drop["comment"] = df_drop["comment"].map(preprocessing_comment)
    if data_train == True:
        df_drop = df_drop[(df_drop["comment"].str.strip() != "") & (df_drop["comment"].str.len() > 1)]
    df_drop.to_csv(csv_result_path, sep=",", index=False)
    print(f"Done processing data in {time.time()-t1}s")

def tokenize(comments):
    '''

    :param comments:
    :return: tokens_comments :list : Mảng các comment mà mỗi comment duoc split thành các từ
    '''
    tokens_comments = []
    for comment in comments:
        comment = preprocessing_comment(comment)
        tokens = [token for token in comment.split(" ")]
        tokens_comments.append(tokens)
    return tokens_comments


def make_embedding(texts_splited, embedding_path, max_features=DEFAULT_MAX_FEATURES):
    '''

    :param texts_splited: các comment đã được tách từ
    :param embedding_path:
    :param max_features:
    :return:
    '''
    embedding_index = KeyedVectors.load_word2vec_format(
            embedding_path, binary=True)
    mean_embedding = np.mean(embedding_index.vectors, axis=0)

    embed_size = mean_embedding.shape[0]
    word_index = sorted(list({word for sentence in texts_splited for word in sentence}))   #dictionary
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
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


def texts_to_sequences(texts_splited, word_map, max_len=DEFAULT_MAX_LENGTH):
        '''
        Chuyển từ thành số  = list(word) thành list(id) tương ứng trong embedding
        :param texts:
        :param word_map:
        :param max_len:
        :return: texts_id shape (n_comment, max_len)
        '''
        texts_id = []
        for sentence in texts_splited:
            sentence = [word_map[word.lower()] for word in sentence][:max_len]
            padded_sentence = np.pad(sentence, (0, max(0, max_len - len(sentence))), 'constant', constant_values=0)
            texts_id.append(padded_sentence)
        return np.array(texts_id)
#
#
# def word_averaging(embedding_matrix, text_id):
#     '''
#     Tạo vector đặc trưng cho comment
#      = Trung bình các vector tạo ra từ các từ trong câu
#     :param embedding_matrix:
#     :param text_id:
#     :return:
#     '''
#     embedding_vectors = []
#     for word_id in text_id:
#         embedding_vectors.append(embedding_matrix[word_id])
#     mean_vector = np.mean(embedding_vectors, axis=0)
#     return mean_vector
#
#
# def word_averaging_list(embedding_matrix, texts_ids):
#     return np.vstack([word_averaging(embedding_matrix, post) for post in texts_ids])


# def create_feature_w2v():
#     comments, labels = read_data()
#
#     tokens_comments = tokenize(comments)
#     embed_size, word_map, embedding_matrix = make_embedding(tokens_comments, w2v_model_path)
#     texts_ids = texts_to_sequences(tokens_comments, word_map)
#
#     X_train = word_averaging_list(embedding_matrix, texts_ids)
#     np.save("features_2.npy", X_train)
#
#     labels = np.asarray(labels)
#     labels = np.reshape(labels,(len(labels),))
#     np.save("labels.npy", labels)

    # return X_train, labels


# def create_feature_tfidf():
#     comments, labels = read_data()
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(comments)
#     X = X.toarray()
#     np.save("features_tfidf.npy", X)
#     labels = np.asarray(labels)
#     labels = np.reshape(labels, (len(labels),))
#     np.save("labels_tfidf.npy", labels)
#
#
if __name__ == "__main__":
    processing_data("data/data_all.csv", "data/data_all_processed.csv")
    # preprocessing_comment(
    #     '"em nhận được hàng rồi rất đẹp 😍😍😍')
