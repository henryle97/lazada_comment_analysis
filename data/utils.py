import pandas as pd


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
    print(df["comment"][:20])


if __name__ == "__main__":
    csv_path = "/home/hisiter/IT/4_year_1/Intro_ML/sentiment_classification/data/data_van.csv"
    csv__res_path = "/home/hisiter/IT/4_year_1/Intro_ML/sentiment_classification/data/data_van_clean.csv"
    # txt2csv("train.txt", "train.csv")
    # test_csv(csv_path)
    clean_csv(csv_path, csv__res_path)