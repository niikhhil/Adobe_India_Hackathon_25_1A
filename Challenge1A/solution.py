import fitz
import pandas as pd
import joblib
import json
import re
import os
import sys 
import time

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
                        x0, top, x1, bottom = [round(c, 2) for c in current_segment_bbox]
                        
                        # --- Reverted to original features ---
                        indentation = round(max(0, x0 - 72.0), 2)
                        word_length = len(current_text_segment.split())
                        char_length = len(current_text_segment)
                        is_all_caps = 1 if current_text_segment.strip().isupper() and any(c.isalpha() for c in current_text_segment.strip()) else 0
                        has_bullet_or_number = 1 if re.match(r'^\s*[-*•–—]\s+|^(\d+\.|\([a-zA-Z0-9]+\)|\w+\))\s+', current_text_segment.strip()) else 0
                        relative_x0 = round(x0 / page_width, 4) if page_width > 0 else 0
                        relative_top = round(top / page_height, 4) if page_height > 0 else 0
                        line_spacing_above = round(top - prev_line_bottom, 2) if prev_line_bottom is not None else 0.0
                        
                        features_list.append({
                            'page': page_num, 'text': current_text_segment.strip(), 'font_size': round(current_segment_font_size, 2),
                            'font_weight': current_segment_font_weight, 'font_name': current_segment_font_name, 'x0': x0, 'top': top,
                            'indentation': indentation, 'word_length': word_length, 'char_length': char_length, 'is_all_caps': is_all_caps,
                            'has_bullet_or_number': has_bullet_or_number, 'relative_x0': relative_x0, 'relative_top': relative_top,
                            'line_spacing_above': line_spacing_above, 'label': ''
                        })
                        prev_line_bottom = bottom
                    
                    current_text_segment = span['text']
                    current_segment_font_weight = span_font_weight
                    current_segment_font_size = span['size']
                    current_segment_font_name = span['font']
                    current_segment_bbox = list(span['bbox'])

            if current_text_segment.strip():
                x0, top, x1, bottom = [round(c, 2) for c in current_segment_bbox]

                indentation = round(max(0, x0 - 72.0), 2)
                word_length = len(current_text_segment.split())
                char_length = len(current_text_segment)
                is_all_caps = 1 if current_text_segment.strip().isupper() and any(c.isalpha() for c in current_text_segment.strip()) else 0
                has_bullet_or_number = 1 if re.match(r'^\s*[-*•–—]\s+|^(\d+\.|\([a-zA-Z0-9]+\)|\w+\))\s+', current_text_segment.strip()) else 0
                relative_x0 = round(x0 / page_width, 4) if page_width > 0 else 0
                relative_top = round(top / page_height, 4) if page_height > 0 else 0
                line_spacing_above = round(top - prev_line_bottom, 2) if prev_line_bottom is not None else 0.0
                
                features_list.append({
                    'page': page_num, 'text': current_text_segment.strip(), 'font_size': round(current_segment_font_size, 2),
                    'font_weight': current_segment_font_weight, 'font_name': current_segment_font_name, 'x0': x0, 'top': top,
                    'indentation': indentation, 'word_length': word_length, 'char_length': char_length, 'is_all_caps': is_all_caps,
                    'has_bullet_or_number': has_bullet_or_number, 'relative_x0': relative_x0, 'relative_top': relative_top,
                    'line_spacing_above': line_spacing_above, 'label': ''
                })
                prev_line_bottom = bottom

    document.close()
    df = pd.DataFrame(features_list)
    return df

def predict_and_generate_json(pdf_input_path, model_path, json_output_path):
    try:
        clf = joblib.load(model_path)
    except Exception:
        return

    df_features = extract_features_from_pdf(pdf_input_path)
    if df_features.empty:
        return

    # --- Reverted to original feature columns ---
    feature_cols = [
        'font_size', 'font_weight', 'x0', 'top', 'indentation',
        'word_length', 'char_length', 'is_all_caps', 'has_bullet_or_number',
        'relative_x0', 'relative_top', 'line_spacing_above', 'page'
    ]
    
    X_predict = df_features[feature_cols]
    if 'font_name' in df_features.columns and hasattr(clf, 'feature_names_in_') and any(col.startswith('font_name_') for col in clf.feature_names_in_):
        X_predict = pd.get_dummies(df_features[feature_cols + ['font_name']], columns=['font_name'], drop_first=True)
    
    if hasattr(clf, 'feature_names_in_'):
        trained_cols = clf.feature_names_in_
        X_predict = X_predict.reindex(columns=trained_cols, fill_value=0)

    predictions = clf.predict(X_predict)
    df_features['predicted_label'] = predictions

    # --- HYBRID HEURISTIC LOGIC (including multi-line title) ---
    document_title = ""
    df_page0 = df_features[df_features['page'] == 0]
    if not df_page0.empty:
        title_candidate_idx = df_page0['font_size'].idxmax()
        title_candidate_row = df_features.loc[title_candidate_idx]
        potential_title_text = title_candidate_row['text'].strip()
        is_meaningful = bool(re.search('[a-zA-Z0-9]', potential_title_text))

        if is_meaningful:
            document_title = potential_title_text
            df_features.loc[title_candidate_idx, 'predicted_label'] = 'title'

            last_title_idx = title_candidate_idx
            for i in range(title_candidate_idx + 1, len(df_features)):
                next_row = df_features.loc[i]
                
                is_same_style = (
                    next_row['page'] == title_candidate_row['page'] and
                    next_row['font_size'] == title_candidate_row['font_size'] and
                    next_row['font_weight'] == title_candidate_row['font_weight'] and
                    next_row['line_spacing_above'] < (title_candidate_row['font_size'] * 1.5)
                )

                if is_same_style:
                    document_title += " " + next_row['text'].strip()
                    df_features.loc[i, 'predicted_label'] = 'title' 
                    last_title_idx = i
                else:
                    break
            
            other_titles_idx = df_features[
                (df_features['predicted_label'] == 'title') &
                (df_features.index > last_title_idx)
            ].index
            df_features.loc[other_titles_idx, 'predicted_label'] = 'H1'
        else:
            if df_features.loc[title_candidate_idx, 'predicted_label'] == 'title':
                df_features.loc[title_candidate_idx, 'predicted_label'] = 'other'

    headings = df_features[df_features['predicted_label'].str.contains(r'^H[1-3]$|title', regex=True)].copy()
    headings = headings.sort_values(by=['page', 'top']).to_dict('records')
    
    last_level = 0
    for i in range(len(headings)):
        current_label = headings[i]['predicted_label']
        if current_label == 'title':
            current_level = 0
        else:
            try:
                current_level = int(current_label[1:])
            except (ValueError, IndexError):
                continue
        if current_level > last_level + 1:
            new_level = last_level + 1
            headings[i]['predicted_label'] = f"H{new_level}"
            current_level = new_level
        last_level = current_level

    for heading in headings:
        original_index = df_features[(df_features['page'] == heading['page']) & (df_features['top'] == heading['top']) & (df_features['text'] == heading['text'])].index
        if not original_index.empty:
            df_features.loc[original_index[0], 'predicted_label'] = heading['predicted_label']

    outline = []
    final_headings = df_features[df_features['predicted_label'].str.match(r'^H[1-3]$')].sort_values(by=['page', 'top'])
    for _, row in final_headings.iterrows():
        outline.append({
            "level": str(row['predicted_label']).strip(),
            "text": row['text'].strip(),
            "page": int(row['page'])
        })

    final_json_output = {"title": document_title, "outline": outline}
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, ensure_ascii=False, indent=4)
    print(f"JSON file for {os.path.basename(pdf_input_path)} successfully created!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python solution.py <input_directory> <output_directory>")
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
            
            start_time = time.time()
            predict_and_generate_json(pdf_input_path, trained_model_path, json_output_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing time for {os.path.basename(pdf_input_path)}: {elapsed_time:.2f} seconds")

