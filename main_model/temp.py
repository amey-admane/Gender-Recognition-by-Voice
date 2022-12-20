import pandas as pd

def r():
    df = pd.read_csv("voice.csv")
    df['label']  = [1 if i == "male" else 0 for i in df.label]
    df.label.value_counts()

    print(df)
    return df