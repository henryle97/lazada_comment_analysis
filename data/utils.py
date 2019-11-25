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
    print(df["comment"])


if __name__ == "__main__":
    csv_path = "train.csv"
    # txt2csv("train.txt", "train.csv")
    test_csv(csv_path)