import fitz  # PyMuPDF
import pandas as pd
import re
import os # Added for path manipulation

def extract_features_from_pdf(pdf_path, output_csv_path):
    """
    Extracts comprehensive text features from a PDF and saves them to a CSV file.
    Includes new features like character length, capitalization, bullet/number presence,
    relative positions, line spacing, and font name.
    This version includes logic to group fragmented text lines that share the same
    vertical position, treating them as a single logical line for feature extraction.
    It also robustly reconstructs text from spans to avoid 'KeyError: text'.

    Args:
        pdf_path (str): The path to the input PDF file.
        output_csv_path (str): The path where the output CSV will be saved.
    """
    document = fitz.open(pdf_path)
    features_list = []

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        # Get page dimensions for relative positioning
        page_width = page.rect.width
        page_height = page.rect.height

        # Collect all lines from all text blocks on the current page
        all_lines_on_page = []
        for block in blocks:
            if block['type'] == 0:  # Only consider text blocks
                for line in block['lines']:
                    all_lines_on_page.append(line)
        
        # Sort all lines on the page by their vertical position (top) then horizontal (x0)
        all_lines_on_page.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))

        grouped_visual_lines = []
        current_visual_line_group = []
        
        # Define a small tolerance for 'top' coordinate to group lines that are visually aligned
        # but might have slight floating point differences. A tolerance of 0.5 is usually good.
        top_tolerance = 0.5 

        for line in all_lines_on_page:
            if not current_visual_line_group:
                current_visual_line_group.append(line)
            else:
                # Check if the current line is a continuation of the previous visual line
                # by comparing its 'top' coordinate with the first line in the current group.
                if abs(line['bbox'][1] - current_visual_line_group[0]['bbox'][1]) < top_tolerance:
                    current_visual_line_group.append(line)
                else:
                    # New visual line starts, process the accumulated current_visual_line_group
                    grouped_visual_lines.append(current_visual_line_group)
                    current_visual_line_group = [line]
        
        if current_visual_line_group: # Add the last accumulated visual line group
            grouped_visual_lines.append(current_visual_line_group)

        prev_visual_line_bottom = None # To calculate line spacing for the reconstructed visual lines

        # Now process each reconstructed visual line group
        for visual_line_group in grouped_visual_lines:
            # Aggregate properties for the entire visual line
            texts_from_line_parts = [] # Collect reconstructed text for each line_part
            font_sizes = []
            font_weights = []
            font_names = []
            
            # Use the bbox of the first line in the group for x0 and top,
            # and calculate the combined bottom from all lines in the group.
            first_line_bbox = visual_line_group[0]['bbox']
            x0 = round(first_line_bbox[0], 2)
            top = round(first_line_bbox[1], 2)
            
            combined_bottom = max(l['bbox'][3] for l in visual_line_group)
            bottom = round(combined_bottom, 2)

            # Collect text and font properties from all parts of the visual line
            for line_part in visual_line_group:
                # Reconstruct text for this specific line_part from its spans
                # This is more robust than relying on line_part['text'] directly,
                # which can sometimes be missing or incomplete.
                line_part_text = "".join([span['text'] for span in line_part['spans']])
                texts_from_line_parts.append(line_part_text)

                for span in line_part['spans']:
                    font_sizes.append(span['size'])
                    font_names.append(span['font'])
                    if "bold" in span['font'].lower() or (span['flags'] & 16): # flag 16 means bold
                        font_weights.append(700) # Common value for bold
                    else:
                        font_weights.append(400) # Common value for regular
            
            # Join the collected text parts (each representing a full line_part's text)
            # Use a space to separate text from different line_parts within the same visual line
            full_line_text = " ".join(texts_from_line_parts)

            if not full_line_text.strip(): # Skip empty reconstructed lines
                continue

            # --- Feature Calculations for the Reconstructed Visual Line ---
            indentation = round(max(0, x0 - 72.0), 2) # Assuming typical PDF margin is around 72 units (1 inch)
            avg_font_size = round(sum(font_sizes) / len(font_sizes), 2) if font_sizes else 0
            
            most_common_font_weight = 400
            if font_weights:
                bold_count = font_weights.count(700)
                regular_count = font_weights.count(400)
                if bold_count > regular_count:
                    most_common_font_weight = 700
                # If counts are equal or only regular, it remains 400
                # If no font_weights (shouldn't happen if full_line_text is not empty), it remains 400

            most_common_font_name = max(set(font_names), key=font_names.count) if font_names else ""
            word_length = len(full_line_text.split())
            char_length = len(full_line_text) 

            is_all_caps = 1 if full_line_text.strip().isupper() and len(full_line_text.strip()) > 1 else 0

            # Check for bullet points or numbered lists using regex
            has_bullet_or_number = 0
            if re.match(r'^\s*[-*•–—]\s+|^(\d+\.|\([a-zA-Z0-9]+\)|\w+\))\s+', full_line_text.strip()):
                has_bullet_or_number = 1

            # Relative positions
            relative_x0 = round(x0 / page_width, 4) if page_width > 0 else 0
            relative_top = round(top / page_height, 4) if page_height > 0 else 0

            # Line spacing above
            line_spacing_above = 0.0
            if prev_visual_line_bottom is not None:
                line_spacing_above = round(top - prev_visual_line_bottom, 2)
            
            # Update prev_visual_line_bottom for the next reconstructed visual line
            prev_visual_line_bottom = bottom

            features = {
                'page': page_num,
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
                'line_spacing_above': line_spacing_above,
                'label': '' # Placeholder for manual labeling
            }
            features_list.append(features)

    document.close()
    
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Features extracted and saved to {output_csv_path}")

if __name__ == "__main__":
    # Ensure this PDF file exists and is accessible
    pdf_file = "D:\Adobe\Pdfs\TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"  
    output_csv = "extracted_features_for_training_11.csv"
    
    # Check if the PDF file exists before proceeding
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at '{pdf_file}'. Please check the path.")
    else:
        extract_features_from_pdf(pdf_file, output_csv)
