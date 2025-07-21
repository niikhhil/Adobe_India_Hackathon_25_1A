import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE

def train_logistic_regression(labeled_csv_path, model_output_path):
    df = pd.read_csv(labeled_csv_path)

    print(f"Initial DataFrame shape: {df.shape}")
    df.info()

    df = df.dropna(subset=['label'])
    df = df[df['label'] != '']

    print(f"\nDataFrame shape after cleaning labels: {df.shape}")
    df.info()

    if df.empty:
        print("\nError: DataFrame is empty after cleaning labels. Cannot proceed with training.")
        return

    feature_cols = ['font_size', 'font_weight', 'x0', 'top', 'indentation', 'word_length',
                    'char_length', 'is_all_caps', 'has_bullet_or_number', 'relative_x0', 'relative_top', 'line_spacing_above', 'page']
    X = df[feature_cols]
    y = df['label']

    print("\nOriginal dataset label distribution (before train-test split):")
    print(y.value_counts())

    # Re-enabling stratify=y now that 'title' and 'H1' have more than 1 instance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)


    print("\nTraining set label distribution after split (stratified):")
    print(y_train.value_counts())

    # Applying SMOTE with k_neighbors=1, as 'title' has only 3 instances (3-1=2, so k_neighbors=1 is safe).
    # The default sampling_strategy='not majority' will work now.
    smote = SMOTE(random_state=42, k_neighbors=1)
    
    print("\nApplying SMOTE to the training data...")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nResampled training set label distribution:")
    print(y_train_resampled.value_counts())


    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
    clf.fit(X_train_resampled, y_train_resampled)

    y_pred = clf.predict(X_test)
    print("\nClassification Report for Logistic Regression (after SMOTE):\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, model_output_path)
    print(f"\nModel saved to: {model_output_path}")

if __name__ == "__main__":
    labeled_csv = "D:\Adobe\csv\extracted_features_for_training_7.csv"
    saved_model = "D:\Adobe\code\myModel.joblib"

    train_logistic_regression(labeled_csv, saved_model)