import pandas as pd


def read_data(stage):
    if stage == 'adoptation':
        return pd.read_csv("./Files/datasets/AraMed/Train.csv"), pd.read_csv("./Files/datasets/AraMed/Test.csv")
    elif stage == 'finetuning':
        return pd.read_csv("./Files/datasets/MedAraBench/Train.csv"), pd.read_csv("./Files/datasets/MedAraBench/Test.csv")