import pandas as pd

def read_file(file_path):
    df = pd.read_csv(file_path, sep=",")
    return df

def txt2csv(txt_path, csv_path):
    comments = []
    labels = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            label, comment = line[0], line[2:]
            comments.append(comment)
            labels.append(label)

    data_dict = {}
    data_dict["stars"] = labels
    data_dict["comment"] = comments
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_path, sep=",", index=False)


def test_csv(csv_path):
    df = pd.read_csv(csv_path, sep=",", usecols=range(2))
    print(df.shape)
    print(df["stars"].value_counts())
    df = df.sort_values(by=["stars"])
    df.to_csv("train_processed_sorted.csv", sep=",", index=False)


if __name__ == "__main__":
    csv_path = "/home/hisiter/IT/4_year_1/Intro_ML/lazada_comment_analysis/data/train_processed.csv"
    csv__res_path = "/home/hisiter/IT/4_year_1/Intro_ML/sentiment_classification/data/data_van_clean.csv"
    # txt2csv("train.txt", "train.csv")
    test_csv(csv_path)

'''

của mình
5    25967
4     2258
1     1773
3     1303
2      803

Của thầy 
5    275
4    119
3     61
1     31
2     14
'''