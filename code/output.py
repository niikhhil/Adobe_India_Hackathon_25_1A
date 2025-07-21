import fitz  # PyMuPDF
import pandas as pd
import joblib
import json
import re

def extract_features_from_pdf_for_prediction(pdf_path):
    document = fitz.open(pdf_path)
    features_list = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        page_width = page.rect.width
        page_height = page.rect.height

        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))

        prev_line_top = None

        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    x0 = round(line['bbox'][0], 2)
                    top = round(line['bbox'][1], 2)
                    indentation = round(max(0, x0 - 72.0), 2)  # 72pts = 1 inch margin approx

                    full_line_text = ""
                    font_sizes = []
                    font_weights = []
                    font_names = []

                    for span in line['spans']:
                        text = span['text']
                        full_line_text += text
                        font_sizes.append(span['size'])
                        font_names.append(span['font'])
                        
                        # Approximate bold detection: font name contains "bold" or flags bit (16)
                        if "bold" in span['font'].lower() or (span['flags'] & 16):
                            font_weights.append(700)
                        else:
                            font_weights.append(400)
                    
                    if not full_line_text.strip():
                        continue

                    avg_font_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0
                    
                    most_common_font_weight = 400
                    if font_weights:
                        bold_count = font_weights.count(700)
                        regular_count = font_weights.count(400)
                        most_common_font_weight = 700 if bold_count > regular_count else 400

                    word_length = len(full_line_text.split())
                    char_length = len(full_line_text)
                    is_all_caps = int(full_line_text.strip().isupper() and bool(re.search(r'[A-Z]', full_line_text.strip())))
                    
                    # Detect bullet or numbered list items
                    has_bullet_or_number = int(bool(re.match(r'^\s*(\d+\.|\*|\-|\•|\u2022|\u25CF)\s+', full_line_text.strip())))

                    relative_x0 = round(x0 / page_width, 4) if page_width else 0
                    relative_top = round(top / page_height, 4) if page_height else 0

                    line_spacing_above = round(top - prev_line_top, 2) if prev_line_top is not None else 0

                    most_common_font_name = max(set(font_names), key=font_names.count) if font_names else "Unknown"

                    features = {
                        'page': page_num + 1,
                        'text': full_line_text.strip(),
                        'font_size': avg_font_size,
                        'font_weight': most_common_font_weight,
                        'font_name': most_common_font_name,
                        'x0': x0,
                        'top': top,
                        'indentation': indentation,
                        'word_length': word_length,
                        'char_length': char_length,
                        'is_all_caps': is_all_caps,
                        'has_bullet_or_number': has_bullet_or_number,
                        'relative_x0': relative_x0,
                        'relative_top': relative_top,
                        'line_spacing_above': line_spacing_above
                    }
                    features_list.append(features)

                    prev_line_top = top

    document.close()
    
    return pd.DataFrame(features_list)

def predict_and_generate_json(pdf_input_path, model_path, json_output_path):
    print(f"Loading model from: {model_path}")
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure your logistic regression model is trained and saved.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Extracting features from: {pdf_input_path}")
    df_features = extract_features_from_pdf_for_prediction(pdf_input_path)

    if df_features.empty:
        print("No text segments extracted from the PDF. Exiting.")
        return
    print(f"Successfully extracted {len(df_features)} text segments.")

    numerical_feature_cols = [
        'font_size', 'font_weight', 'x0', 'top', 'indentation',
        'word_length', 'char_length', 'is_all_caps', 'has_bullet_or_number',
        'relative_x0', 'relative_top', 'line_spacing_above', 'page'
    ]
    categorical_feature_cols = ['font_name']

    if all(col in df_features.columns for col in categorical_feature_cols):
        print(f"\nApplying one-hot encoding to: {categorical_feature_cols}")
        X_predict = pd.get_dummies(df_features[numerical_feature_cols + categorical_feature_cols],
                                   columns=categorical_feature_cols, drop_first=True)
    else:
        X_predict = df_features[numerical_feature_cols]

    if hasattr(clf, 'feature_names_in_'):
        trained_cols = clf.feature_names_in_
        X_predict = X_predict.reindex(columns=trained_cols, fill_value=0)
    else:
        print("Warning: Model missing 'feature_names_in_' attribute. Continuing without explicit feature alignment.")

    print("Predicting labels...")
    predictions = clf.predict(X_predict)

    df_features['predicted_label'] = predictions

    print("\nSample predictions:")
    print(df_features[['text', 'predicted_label']].head(10))

    document_title = ""
    outline = []

    for _, row in df_features.iterrows():
        label = str(row['predicted_label']).strip()
        text = row['text'].strip()
        page = int(row['page'])

        if label == 'title':
            if not document_title:
                document_title = text
        elif label.startswith("H") and len(label) > 1 and label[1:].isdigit():
            outline.append({
                "level": label,
                "text": text,
                "page": page
            })

    final_json_output = {
        "title": document_title,
        "outline": outline
    }

    print(f"\nSaving JSON output to {json_output_path} ...")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, ensure_ascii=False, indent=4)

    print("JSON output generated successfully!")

if __name__ == "__main__":
    # Update these paths before running
    pdf_to_classify = r"D:\\Adobe\\Pdfs\\TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"
    trained_model_path = r"D:\\Adobe\\code\\myModel.joblib"
    output_json_file = r"D:\\Adobe\\code\\pdf_structured_output_custom.json"

    predict_and_generate_json(pdf_to_classify, trained_model_path, output_json_file)
