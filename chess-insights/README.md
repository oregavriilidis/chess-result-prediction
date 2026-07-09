# ♟️ Chess Insights — Ανάλυση & Πρόβλεψη Αποτελεσμάτων Παρτίδων Σκακιού

Ένα end-to-end data science project που αναλύει δεδομένα παρτιδών σκακιού
(rating παικτών, άνοιγμα, time control, διάρκεια) και εκπαιδεύει ένα μοντέλο
Machine Learning που προβλέπει το αποτέλεσμα μιας παρτίδας.

Το project καλύπτει ολόκληρο τον κύκλο ενός data pipeline:
**data generation → cleaning → EDA → feature engineering → modeling → evaluation → CLI inference**.

## 📌 Γιατί αυτό το project

Ως φοιτητής Πληροφορικής και ενεργός σκακιστής, ήθελα να συνδυάσω ένα hobby
που ξέρω καλά με τεχνικές ανάλυσης δεδομένων, ώστε να μπορώ να εξηγήσω
με νόημα κάθε βήμα του pipeline — από τον καθαρισμό δεδομένων μέχρι την
ερμηνεία του μοντέλου.

## 🗂️ Δομή Project

```
chess-insights/
├── data/
│   ├── sample_games.csv       # raw synthetic dataset (3000 παρτίδες)
│   └── clean_games.csv        # καθαρισμένο dataset με engineered features
├── src/
│   ├── generate_data.py       # δημιουργία ρεαλιστικού synthetic dataset
│   ├── data_cleaning.py       # καθαρισμός & feature engineering
│   ├── eda.py                 # exploratory data analysis + γραφήματα
│   ├── train_model.py         # εκπαίδευση Random Forest classifier
│   └── predict.py             # CLI εργαλείο πρόβλεψης
├── tests/
│   └── test_data_cleaning.py  # unit tests (pytest)
├── images/                    # παραγόμενα γραφήματα
├── requirements.txt
└── README.md
```

## 📊 Dataset

Το dataset είναι **synthetic**, αλλά χτισμένο πάνω σε ρεαλιστική στατιστική
λογική (Elo expected score formula), ώστε τα patterns να είναι αντίστοιχα
αυτών σε πραγματικά δεδομένα Lichess. Περιλαμβάνει:

| Στήλη | Περιγραφή |
|---|---|
| `white_rating` / `black_rating` | Elo rating παικτών |
| `opening_name` / `opening_eco` | Άνοιγμα παρτίδας (ECO code) |
| `time_control` | Bullet / Blitz / Rapid / Classical |
| `num_moves` | Διάρκεια παρτίδας σε κινήσεις |
| `result` | Αποτέλεσμα: white / black / draw |

> 💡 Θέλεις πραγματικά δεδομένα; Κατέβασε ένα PGN/CSV dataset από το
> [Lichess Database](https://database.lichess.org/) ή το
> [Kaggle Chess Games Dataset](https://www.kaggle.com/datasets/datasnaek/chess)
> και πέρασέ το στο `data_cleaning.py` — η δομή στηλών είναι συμβατή.

## 🔍 Βασικά Ευρήματα (EDA)

- Το ποσοστό νίκης των λευκών αυξάνεται σχεδόν γραμμικά με τη διαφορά rating,
  όπως αναμένεται από τη θεωρία Elo.
- Ορισμένα ανοίγματα εμφανίζουν ελαφρώς καλύτερο win rate για τα λευκά,
  αλλά η επιρροή του ανοίγματος είναι πολύ μικρότερη από τη διαφορά rating.
- Οι ισοπαλίες είναι σπανιότερες σε bullet/blitz παρτίδες σε σχέση με classical.

Δες όλα τα γραφήματα στον φάκελο [`images/`](images/).

## 🤖 Μοντέλο

Χρησιμοποιείται **Random Forest Classifier** (scikit-learn) με:
- Αριθμητικά features: `rating_diff`, `avg_rating`, `num_moves`
- Categorical features (One-Hot Encoded): `opening_name`, `time_control`

**Αποτελέσματα στο test set:**
- Accuracy: ~58% (σε πρόβλημα 3 κλάσεων, το baseline του "πάντα η πιο συχνή κλάση" είναι ~35%)
- Το μοντέλο προβλέπει καλύτερα τα white/black wins απ' ό,τι τα draws (αναμενόμενο —
  οι ισοπαλίες είναι εγγενώς πιο δύσκολο να προβλεφθούν)

Confusion matrix & feature importance: [`images/confusion_matrix.png`](images/confusion_matrix.png),
[`images/feature_importance.png`](images/feature_importance.png)

## 🚀 Πώς να το τρέξεις

```bash
# 1. Κλωνοποίησε το repo
git clone https://github.com/<username>/chess-insights.git
cd chess-insights

# 2. Δημιούργησε virtual environment (προαιρετικό αλλά συνιστάται)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Εγκατάστησε τα dependencies
pip install -r requirements.txt

# 4. Τρέξε το πλήρες pipeline
python src/generate_data.py     # δημιουργεί data/sample_games.csv
python src/data_cleaning.py     # δημιουργεί data/clean_games.csv
python src/eda.py               # παράγει τα γραφήματα στο images/
python src/train_model.py       # εκπαιδεύει το μοντέλο

# 5. Δοκίμασε μια πρόβλεψη
python src/predict.py --white 1900 --black 1500 \
    --opening "Ruy Lopez" --time-control Rapid --moves 45
```

Παράδειγμα εξόδου:
```
Πρόβλεψη: white
Πιθανότητες ανά αποτέλεσμα:
   white:  54.7%
    draw:  24.2%
   black:  21.1%
```

## ✅ Tests

```bash
pytest tests/ -v
```

## 🛠️ Τεχνολογίες

Python 3.12 · pandas · numpy · scikit-learn · matplotlib · seaborn · joblib · pytest

## 📈 Πιθανές Επεκτάσεις

- Ενσωμάτωση πραγματικών δεδομένων μέσω του [Lichess API](https://lichess.org/api)
- Ανάλυση σε επίπεδο κίνησης (move-by-move) με python-chess για εντοπισμό λαθών/blunders
- Hyperparameter tuning (GridSearchCV) και σύγκριση με XGBoost/LightGBM
- Web dashboard (Streamlit) για interactive εξερεύνηση των δεδομένων

## 📄 Άδεια Χρήσης

Αυτό το project διανέμεται υπό την [MIT License](LICENSE).

---

*Δημιουργήθηκε ως portfolio project από φοιτητή Πληροφορικής 4ου έτους.*
