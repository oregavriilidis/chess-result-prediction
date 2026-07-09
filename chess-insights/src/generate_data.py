"""
generate_data.py
-----------------
Δημιουργεί ένα ρεαλιστικό (synthetic) dataset παρτίδων σκακιού, στη λογική
του δημόσιου Lichess dataset (https://database.lichess.org/).

Σημείωση: Αυτό το script χρησιμοποιείται ώστε το project να τρέχει αυτόνομα,
χωρίς να χρειάζεται download μεγάλου αρχείου. Αν θέλεις να δουλέψεις με
πραγματικά δεδομένα, κατέβασε ένα PGN dataset από:
  - https://database.lichess.org/
  - https://www.kaggle.com/datasets/datasnaek/chess
και πέρασέ το μέσα από το data_cleaning.py (η δομή στηλών είναι συμβατή).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

OPENINGS = [
    ("Sicilian Defense", "B20"),
    ("Italian Game", "C50"),
    ("Ruy Lopez", "C60"),
    ("Queen's Gambit", "D06"),
    ("French Defense", "C00"),
    ("Caro-Kann Defense", "B10"),
    ("King's Indian Defense", "E60"),
    ("English Opening", "A10"),
    ("Scandinavian Defense", "B01"),
    ("London System", "D02"),
    ("Slav Defense", "D10"),
    ("Pirc Defense", "B07"),
]

TIME_CONTROLS = ["Bullet", "Blitz", "Rapid", "Classical"]
TIME_CONTROL_WEIGHTS = [0.30, 0.40, 0.20, 0.10]


def sample_rating(n):
    # Ρεαλιστική κατανομή ratings (κανονική γύρω από 1500, όπως στο Lichess)
    return np.clip(RNG.normal(1500, 300, n), 600, 2800).round().astype(int)


def simulate_result(rating_diff, opening_bias, rng):
    """
    Πιθανότητα νίκης του λευκού βάσει διαφοράς rating (λογιστική συνάρτηση,
    προσομοιώνει Elo expected score) + μικρή τυχαία επιρροή ανοίγματος.
    """
    expected_white = 1 / (1 + 10 ** (-(rating_diff + opening_bias) / 400))
    draw_prob = np.clip(0.12 - abs(rating_diff) / 4000, 0.03, 0.15)
    r = rng.random()
    if r < draw_prob:
        return "draw"
    remaining = 1 - draw_prob
    if rng.random() < expected_white / (expected_white + (1 - expected_white)):
        return "white" if rng.random() < expected_white else "black"
    return "white" if rng.random() < expected_white else "black"


def generate(n_games: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    white_ratings = sample_rating(n_games)
    rating_spread = rng.normal(0, 150, n_games).round().astype(int)
    black_ratings = np.clip(white_ratings - rating_spread, 600, 2800)

    opening_idx = rng.integers(0, len(OPENINGS), n_games)
    opening_names = [OPENINGS[i][0] for i in opening_idx]
    opening_eco = [OPENINGS[i][1] for i in opening_idx]
    # Κάθε άνοιγμα έχει μια μικρή "προτίμηση" υπέρ λευκών/μαύρων
    opening_bias_map = {name: rng.normal(0, 25) for name, _ in OPENINGS}

    time_controls = rng.choice(TIME_CONTROLS, size=n_games, p=TIME_CONTROL_WEIGHTS)

    results = []
    n_moves = []
    for i in range(n_games):
        diff = int(white_ratings[i] - black_ratings[i])
        bias = opening_bias_map[opening_names[i]]
        res = simulate_result(diff, bias, rng)
        results.append(res)
        base_moves = rng.integers(20, 70)
        if res == "draw":
            base_moves += rng.integers(0, 20)
        n_moves.append(int(base_moves))

    df = pd.DataFrame({
        "white_rating": white_ratings,
        "black_rating": black_ratings,
        "opening_name": opening_names,
        "opening_eco": opening_eco,
        "time_control": time_controls,
        "num_moves": n_moves,
        "result": results,  # white / black / draw
    })
    return df


if __name__ == "__main__":
    df = generate(3000)
    df.to_csv("data/sample_games.csv", index=False)
    print(f"Δημιουργήθηκαν {len(df)} παρτίδες -> data/sample_games.csv")
    print(df.head())
