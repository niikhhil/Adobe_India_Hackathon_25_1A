import fitz
import pandas as pd
import joblib
import json
import re
import os
import sys # Import sys to access command-line arguments
import time # Import the time module to measure execution time

def extract_features_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    features_list = [] 

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]       
        page_width = page.rect.width
        page_height = page.rect.height 

        all_lines_on_page = []
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    all_lines_on_page.append(line)       
        all_lines_on_page.sort(key=lambda l: (l['bbox'][1], l['bbox'][0])) 

        grouped_visual_lines = []
        current_visual_line_group = []       
        top_tolerance = 0.5 

        for line in all_lines_on_page:
            if not current_visual_line_group:
                current_visual_line_group.append(line)
            else:
                if abs(line['bbox'][1] - current_visual_line_group[0]['bbox'][1]) < top_tolerance:
                    current_visual_line_group.append(line)
                else:
                    grouped_visual_lines.append(current_visual_line_group)
                    current_visual_line_group = [line]       
        if current_visual_line_group:
            grouped_visual_lines.append(current_visual_line_group) 

        prev_line_bottom = None 

        for visual_line_group in grouped_visual_lines:
            all_spans_in_group = []
            for line_part in visual_line_group:
                for span in line_part['spans']:
                    all_spans_in_group.append(span)           
            all_spans_in_group.sort(key=lambda s: s['bbox'][0]) 

            current_text_segment = ""
            current_segment_font_weight = None
            current_segment_font_size = None
            current_segment_font_name = None
            current_segment_bbox = None 

            for i, span in enumerate(all_spans_in_group):
                span_font_weight = 700 if (span['flags'] & 16) or "bold" in span['font'].lower() or "black" in span['font'].lower() or "heavy" in span['font'].lower() else 400             
                if current_segment_font_weight is None:
                    current_segment_font_weight = span_font_weight
                    current_segment_font_size = span['size']
                    current_segment_font_name = span['font']
                    current_segment_bbox = list(span['bbox']) 
                    current_text_segment = span['text']
                elif span_font_weight == current_segment_font_weight:
                    current_text_segment += span['text']
                    current_segment_bbox[2] = max(current_segment_bbox[2], span['bbox'][2]) 
                    current_segment_bbox[3] = max(current_segment_bbox[3], span['bbox'][3]) 
                    current_segment_bbox[0] = min(current_segment_bbox[0], span['bbox'][0]) 
                    current_segment_bbox[1] = min(current_segment_bbox[1], span['bbox'][1]) 

                else:
                    if current_text_segment.strip(): 
                        x0 = round(current_segment_bbox[0], 2)
                        top = round(current_segment_bbox[1], 2)
                        bottom = round(current_segment_bbox[3], 2) 

                        indentation = round(max(0, x0 - 72.0), 2)
                        word_length = len(current_text_segment.split())
                        char_length = len(current_text_segment)
                        is_all_caps = 1 if current_text_segment.strip().isupper() and any(c.isalpha() for c in current_text_segment.strip()) else 0                     
                        has_bullet_or_number = 0
                        if re.match(r'^\s*[-*•–—]\s+|^(\d+\.|\([a-zA-Z0-9]+\)|\w+\))\s+', current_text_segment.strip()):
                            has_bullet_or_number = 1 

                        relative_x0 = round(x0 / page_width, 4) if page_width > 0 else 0
                        relative_top = round(top / page_height, 4) if page_height > 0 else 0 

                        line_spacing_above = 0.0
                        if prev_line_bottom is not None:
                            line_spacing_above = round(top - prev_line_bottom, 2)                    
                        features_list.append({
                            'page': page_num,
                            'text': current_text_segment.strip(),
                            'font_size': round(current_segment_font_size, 2),
                            'font_weight': current_segment_font_weight,
                            'font_name': current_segment_font_name,
                            'x0': x0,
                            'top': top,
                            'indentation': indentation,
                            'word_length': word_length,
                            'char_length': char_length,
                            'is_all_caps': is_all_caps,
                            'has_bullet_or_number': has_bullet_or_number,
                            'relative_x0': relative_x0,
                            'relative_top': relative_top,
                            'line_spacing_above': line_spacing_above,
                            'label': ''
                        })
                        prev_line_bottom = bottom 

                    current_text_segment = span['text']
                    current_segment_font_weight = span_font_weight
                    current_segment_font_size = span['size']
                    current_segment_font_name = span['font']
                    current_segment_bbox = list(span['bbox']) 

            if current_text_segment.strip():
                x0 = round(current_segment_bbox[0], 2)
                top = round(current_segment_bbox[1], 2)
                bottom = round(current_segment_bbox[3], 2) 

                indentation = round(max(0, x0 - 72.0), 2)
                word_length = len(current_text_segment.split())
                char_length = len(current_text_segment)
                is_all_caps = 1 if current_text_segment.strip().isupper() and any(c.isalpha() for c in current_text_segment.strip()) else 0             
                has_bullet_or_number = 0
                if re.match(r'^\s*[-*•–—]\s+|^(\d+\.|\([a-zA-Z0-9]+\)|\w+\))\s+', current_text_segment.strip()):
                    has_bullet_or_number = 1 

                relative_x0 = round(x0 / page_width, 4) if page_width > 0 else 0
                relative_top = round(top / page_height, 4) if page_height > 0 else 0 

                line_spacing_above = 0.0
                if prev_line_bottom is not None:
                    line_spacing_above = round(top - prev_line_bottom, 2) 

                features_list.append({
                    'page': page_num,
                    'text': current_text_segment.strip(),
                    'font_size': round(current_segment_font_size, 2),
                    'font_weight': current_segment_font_weight,
                    'font_name': current_segment_font_name,
                    'x0': x0,
                    'top': top,
                    'indentation': indentation,
                    'word_length': word_length,
                    'char_length': char_length,
                    'is_all_caps': is_all_caps,
                    'has_bullet_or_number': has_bullet_or_number,
                    'relative_x0': relative_x0,
                    'relative_top': relative_top,
                    'line_spacing_above': line_spacing_above,
                    'label': ''
                })
                prev_line_bottom = bottom 

    document.close()
    
    df = pd.DataFrame(features_list)
    return df 

def predict_and_generate_json(pdf_input_path, model_path, json_output_path):
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        return
    except Exception as e:
        return

    df_features = extract_features_from_pdf(pdf_input_path) 

    if df_features.empty:
        return

    feature_cols = [
        'font_size', 'font_weight', 'x0', 'top', 'indentation',
        'word_length', 'char_length',
        'is_all_caps', 'has_bullet_or_number',
        'relative_x0', 'relative_top', 'line_spacing_above', 'page'
    ]
    if 'font_name' in df_features.columns and hasattr(clf, 'feature_names_in_') and \
       any(col.startswith('font_name_') for col in clf.feature_names_in_):
        X_predict = pd.get_dummies(df_features[feature_cols + ['font_name']], columns=['font_name'], drop_first=True)
    else:
        X_predict = df_features[feature_cols] 

    if hasattr(clf, 'feature_names_in_'):
        trained_cols = clf.feature_names_in_
        X_predict = X_predict.reindex(columns=trained_cols, fill_value=0)
    else:
        pass

    predictions = clf.predict(X_predict)
    df_features['predicted_label'] = predictions 

    document_title = ""
    outline = []
    for _, row in df_features.iterrows():
        label = str(row['predicted_label']).strip()
        text = row['text'].strip()
        page = int(row['page']) 

        if label == 'title' and not document_title:
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

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, ensure_ascii=False, indent=4)
    print(f"JSON file for {os.path.basename(pdf_input_path)} successfully created!")

if __name__ == "__main__":
    if len(sys.argv) != 3: 
        print("Usage: python solution.py <input_directory> <output_directory>")
        print("Example: python solution.py /app/input /app/output")
        sys.exit(1) 

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] 

    trained_model_path = "/app/trained_model.joblib" 

    os.makedirs(output_dir, exist_ok=True) 

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_input_path = os.path.join(input_dir, filename) 
            
            json_output_filename = os.path.splitext(filename)[0] + ".json"
            json_output_path = os.path.join(output_dir, json_output_filename) 
            
            start_time = time.time() # Record start time
            predict_and_generate_json(pdf_input_path, trained_model_path, json_output_path)
            end_time = time.time() # Record end time
            elapsed_time = end_time - start_time # Calculate elapsed time
            print(f"Processing time for {os.path.basename(pdf_input_path)}: {elapsed_time:.2f} seconds") # Print execution time
