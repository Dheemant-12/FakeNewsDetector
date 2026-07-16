import pandas as pd


def load_dataset():
    """
    Load and combine the fake and true news datasets.
    """

    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    return df