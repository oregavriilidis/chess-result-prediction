"""
eda.py
------
Εξερευνητική ανάλυση δεδομένων (EDA) στο καθαρισμένο dataset παρτίδων.
Παράγει γραφήματα στον φάκελο images/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def load_clean(path: str = "data/clean_games.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def plot_result_distribution(df: pd.DataFrame, out="images/result_distribution.png"):
    plt.figure(figsize=(6, 4))
    order = df["result"].value_counts().index
    sns.countplot(data=df, x="result", hue="result", order=order, palette="viridis", legend=False)
    plt.title("Κατανομή Αποτελεσμάτων Παρτίδων")
    plt.xlabel("Αποτέλεσμα")
    plt.ylabel("Πλήθος Παρτίδων")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_winrate_by_rating_diff(df: pd.DataFrame, out="images/winrate_by_rating_diff.png"):
    bins = [-1000, -300, -150, -50, 50, 150, 300, 1000]
    labels = ["<-300", "-300..-150", "-150..-50", "-50..50", "50..150", "150..300", ">300"]
    df = df.copy()
    df["rating_bucket"] = pd.cut(df["rating_diff"], bins=bins, labels=labels)
    winrate = df.groupby("rating_bucket", observed=True)["result"].apply(
        lambda x: (x == "white").mean()
    )
    plt.figure(figsize=(7, 4))
    winrate.plot(kind="bar", color="#4C72B0")
    plt.title("Ποσοστό Νίκης Λευκών ανά Διαφορά Rating (White - Black)")
    plt.xlabel("Διαφορά Rating")
    plt.ylabel("Win rate Λευκών")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_winrate_by_opening(df: pd.DataFrame, out="images/winrate_by_opening.png"):
    winrate = (
        df.groupby("opening_name")["result"]
        .apply(lambda x: (x == "white").mean())
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(8, 5))
    winrate.plot(kind="barh", color="#55A868")
    plt.title("Win rate Λευκών ανά Άνοιγμα")
    plt.xlabel("Win rate Λευκών")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_moves_distribution(df: pd.DataFrame, out="images/moves_distribution.png"):
    plt.figure(figsize=(7, 4))
    sns.histplot(df["num_moves"], bins=30, kde=True, color="#C44E52")
    plt.title("Κατανομή Διάρκειας Παρτίδας (σε κινήσεις)")
    plt.xlabel("Αριθμός Κινήσεων")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_time_control_counts(df: pd.DataFrame, out="images/time_control_counts.png"):
    plt.figure(figsize=(6, 4))
    order = df["time_control"].value_counts().index
    sns.countplot(data=df, x="time_control", hue="time_control", order=order, palette="mako", legend=False)
    plt.title("Πλήθος Παρτίδων ανά Time Control")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def run_all():
    df = load_clean()
    plot_result_distribution(df)
    plot_winrate_by_rating_diff(df)
    plot_winrate_by_opening(df)
    plot_moves_distribution(df)
    plot_time_control_counts(df)
    print("Τα γραφήματα αποθηκεύτηκαν στον φάκελο images/")


if __name__ == "__main__":
    run_all()
