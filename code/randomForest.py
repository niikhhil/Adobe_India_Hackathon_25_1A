import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_model(csv_labeled_path, model_save_path):
    df = pd.read_csv(csv_labeled_path)

    # Drop rows with no label (if any)
    df = df.dropna(subset=['label'])

    feature_cols = ['font_size', 'font_weight', 'x0', 'top', 'indentation', 'word_length', 'page']
    X = df[feature_cols]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_save_path)
    print(f"Model saved to {model_save_path}")
