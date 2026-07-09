"""
data_cleaning.py
-----------------
Καθαρίζει το raw dataset παρτίδων και παράγει επιπλέον features
χρήσιμα για ανάλυση και μοντελοποίηση.
"""

import pandas as pd


def load_raw(path: str = "data/sample_games.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Αφαίρεση τυχόν κενών/ελλιπών γραμμών
    df = df.dropna(subset=["white_rating", "black_rating", "result"])

    # Τύποι δεδομένων
    df["white_rating"] = df["white_rating"].astype(int)
    df["black_rating"] = df["black_rating"].astype(int)
    df["num_moves"] = df["num_moves"].astype(int)

    # Έγκυρα results μόνο
    df = df[df["result"].isin(["white", "black", "draw"])]

    # Feature engineering
    df["rating_diff"] = df["white_rating"] - df["black_rating"]
    df["avg_rating"] = (df["white_rating"] + df["black_rating"]) / 2
    df["higher_rated_won"] = (
        ((df["rating_diff"] > 0) & (df["result"] == "white")) |
        ((df["rating_diff"] < 0) & (df["result"] == "black"))
    )

    return df.reset_index(drop=True)


def save_clean(df: pd.DataFrame, path: str = "data/clean_games.csv") -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    save_clean(cleaned)
    print(f"Καθαρίστηκαν {len(cleaned)} παρτίδες -> data/clean_games.csv")
    print(cleaned.describe(include="all").T)
