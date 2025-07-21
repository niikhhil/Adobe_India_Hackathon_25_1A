import fitz
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
        
        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))

        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    x0 = round(line['bbox'][0], 2)
                    indentation = round(max(0, x0 - 72.0), 2) 

                    full_line_text = ""
                    font_sizes = []
                    font_weights = []
                    
                    for span in line['spans']:
                        text = span['text']
                        full_line_text += text
                        font_sizes.append(span['size'])
                        
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
                        if bold_count > regular_count:
                            most_common_font_weight = 700
                        elif bold_count == 0 and regular_count == 0:
                            most_common_font_weight = 400 
                        else:
                             most_common_font_weight = 400

                    word_length = len(full_line_text.split())

                    features = {
                        'page': page_num + 1,
                        'text': full_line_text.strip(),
                        'font_size': avg_font_size,
                        'font_weight': most_common_font_weight,
                        'x0': round(line['bbox'][0], 2),
                        'top': round(line['bbox'][1], 2),
                        'indentation': indentation,
                        'word_length': word_length
                    }
                    features_list.append(features)

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
        print("No text features extracted from the PDF. Exiting.")
        return
    print(f"Successfully extracted {len(df_features)} text segments.")

    feature_cols = ['font_size', 'font_weight', 'x0', 'top', 'indentation', 'word_length', 'page']

    missing_cols = [col for col in feature_cols if col not in df_features.columns]
    if missing_cols:
        print(f"Error: Missing required feature columns in extracted data: {missing_cols}")
        print("Please ensure your PDF extraction process provides all necessary features.")
        return

    X_predict = df_features[feature_cols]

    print("Predicting labels...")
    predictions = clf.predict(X_predict)
    df_features['predicted_label'] = predictions
    print("\nSample of predictions (first 10 rows):")
    print(df_features[['text', 'predicted_label']].head(10))

    document_title = ""
    outline = []
    
    for index, row in df_features.iterrows():
        label = str(row['predicted_label']).strip() # Ensure label is string and stripped
        text = row['text']
        page = row['page']

        if label == 'title':
            if not document_title: # Take the first detected 'title' as the main document title
                document_title = text.strip()
        elif label.startswith("H") and len(label) > 1 and label[1:].isdigit(): # Check for H1, H2, H3 format
            heading_level_num = label[1:] # e.g., 'H1' -> '1'
            heading_level_str = f"H{heading_level_num}" # This is already the format, but ensures consistency
            
            outline.append({
                "level": heading_level_str,
                "text": text.strip(),
                "page": page
            })
    
    final_json_output = {
        "title": document_title,
        "outline": outline
    }

    print(f"\nDebug: Final JSON content before writing to file:")
    print(f"  Title: '{document_title}'")
    print(f"  Outline entries: {len(outline)}")
    if outline:
        print(f"  First 3 outline entries: {outline[:3]}")

    print(f"Saving results to {json_output_path}")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, ensure_ascii=False, indent=4)
    print("JSON output generated successfully!")


if __name__ == "__main__":
    pdf_to_classify = "D:\Adobe\Pdfs\E0CCG5S312.pdf"
    trained_model_path = "code\myModel.joblib" # Ensure this path is correct
    output_json_file = "pdf_structured_output_custom.json"

    predict_and_generate_json(pdf_to_classify, trained_model_path, output_json_file)