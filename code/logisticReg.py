import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_logistic_regression(labeled_csv_path, model_output_path):
    # Load labeled dataset
    df = pd.read_csv(labeled_csv_path)

    # Keep only rows with valid labels
    df = df.dropna(subset=['label'])
    df = df[df['label'] != '']

    feature_cols = ['font_size', 'font_weight', 'x0', 'top', 'indentation', 'word_length', 'page']
    X = df[feature_cols]
    y = df['label']

    # Split dataset into training and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Create and train Logistic Regression model
    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train, y_train)

    # Predict and evaluate on test set
    y_pred = clf.predict(X_test)
    print("Classification Report for Logistic Regression:\n")
    print(classification_report(y_test, y_pred))

    # Save the trained model to disk
    joblib.dump(clf, model_output_path)
    print(f"\nModel saved to: {model_output_path}")

if __name__ == "__main__":
    labeled_csv = "labeled_pdf_words.csv"      # your labeled CSV file path
    saved_model = "logreg_pdf_headings_model.joblib"
    train_logistic_regression(labeled_csv, saved_model)
