import pandas as pd

def opener_dataframe(PATH:str="../Data/ETHUSDT-5m.zip"):
    df = pd.read_csv(PATH).drop("Unnamed: 0",axis=1)
    df["open_date"] = pd.to_datetime(df["open_date"])
    return df

if __name__ == "__main__":
    PATH = "../Data/ETHUSDT-5m.zip"
    df = opener_dataframe(PATH)