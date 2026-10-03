import pandas as pd
from sklearn.model_selection import train_test_split

# target_colmun = "charges"
def preprocess_data(df,target_colmun):
    df = df.copy()

    df = df.drop_duplicates()

    X = df.drop(columns=[target_colmun])
    y = df[target_colmun]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from data_loading import load_data
    df = load_data()
    preprocess_data(df)