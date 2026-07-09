"""
predict.py
----------
Απλό CLI εργαλείο: δίνεις rating λευκού/μαύρου, άνοιγμα, time control και
αριθμό κινήσεων, και το μοντέλο προβλέπει το πιθανό αποτέλεσμα.

Χρήση:
    python src/predict.py --white 1800 --black 1600 \
        --opening "Sicilian Defense" --time-control Blitz --moves 40
"""

import argparse

import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Πρόβλεψη αποτελέσματος παρτίδας σκακιού")
    parser.add_argument("--white", type=int, required=True, help="Rating λευκού παίκτη")
    parser.add_argument("--black", type=int, required=True, help="Rating μαύρου παίκτη")
    parser.add_argument("--opening", type=str, default="Sicilian Defense", help="Όνομα ανοίγματος")
    parser.add_argument("--time-control", type=str, default="Blitz",
                         choices=["Bullet", "Blitz", "Rapid", "Classical"])
    parser.add_argument("--moves", type=int, default=40, help="Αναμενόμενος αριθμός κινήσεων")
    return parser.parse_args()


def main():
    args = parse_args()
    model = joblib.load("src/chess_result_model.joblib")

    row = pd.DataFrame([{
        "rating_diff": args.white - args.black,
        "avg_rating": (args.white + args.black) / 2,
        "num_moves": args.moves,
        "opening_name": args.opening,
        "time_control": args.time_control,
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.classes_

    print(f"\nΠρόβλεψη: {pred}")
    print("Πιθανότητες ανά αποτέλεσμα:")
    for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
        print(f"  {cls:>6}: {p * 100:5.1f}%")


if __name__ == "__main__":
    main()
