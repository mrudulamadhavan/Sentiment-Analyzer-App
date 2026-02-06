import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocess_review import preprocess

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("product_reviews.csv")

X = df["review_text"].apply(preprocess)
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODELS (FROM IPYNB)
# -------------------------------
models = {
    "naive_bayes": Pipeline([
        ("vectorizer", CountVectorizer(max_features=5000)),
        ("classifier", MultinomialNB())
    ]),
    "logistic_regression": Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "random_forest": Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=5000)),
        ("classifier", RandomForestClassifier(n_estimators=50))
    ])
}

# -------------------------------
# TRAIN & SAVE
# -------------------------------
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"{name} accuracy:", accuracy_score(y_test, preds))

    joblib.dump(model, f"best_models/{name}.pkl")

print("✅ Models trained and saved successfully")
