import pdfplumber
import csv
import re

def estimate_font_weight(fontname: str) -> int:
    """Estimate font weight from the font name."""
    fontname_lower = fontname.lower()
    if "thin" in fontname_lower:
        return 100
    elif "extralight" in fontname_lower or "ultralight" in fontname_lower:
        return 200
    elif "light" in fontname_lower:
        return 300
    elif "book" in fontname_lower:
        return 350
    elif "regular" in fontname_lower or "normal" in fontname_lower:
        return 400
    elif "medium" in fontname_lower:
        return 500
    elif "semibold" in fontname_lower or "demibold" in fontname_lower:
        return 600
    elif "bold" in fontname_lower:
        return 700
    elif "extrabold" in fontname_lower or "ultrabold" in fontname_lower:
        return 800
    elif "black" in fontname_lower or "heavy" in fontname_lower:
        return 900
    else:
        return 400

def calculate_caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase letters in the given text."""
    if not text:
        return 0.0
    uppercase_count = sum(1 for char in text if char.isupper())
    return uppercase_count / len(text)

def extract_features_to_csv(pdf_path, csv_path):
    """Extract sentence-level features from the PDF and save to CSV."""
    with pdfplumber.open(pdf_path) as pdf, open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['page', 'sentence_index', 'text', 'avg_font_size', 'bold',
                      'caps_ratio', 'indentation', 'num_words', 'text_len', 
                      'digit_ratio', 'starts_with_number', 'contains_colon', 'line_height']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True)
            if not text:
                print(f"Warning: No text extracted from page {page_num}")
                continue

            # Split page text into sentences (simple regex)
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Extract words with bounding boxes to map sentences properly
            words = page.extract_words()

            for sentence_idx, sentence in enumerate(sentences):
                # Find words that exist within the sentence text
                sentence_words = [w for w in words if w['text'] in sentence]

                # Collect characters belonging to the words in the sentence
                sentence_chars = []
                for w in sentence_words:
                    chars_in_word = [
                        c for c in page.chars
                        if c['x0'] >= w['x0'] and c['x1'] <= w['x1'] and
                           c['top'] >= w['top'] and c['bottom'] <= w['bottom']
                    ]
                    sentence_chars.extend(chars_in_word)

                # Calculate average font size
                font_sizes = [c['size'] for c in sentence_chars]
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

                # Calculate average font weight
                font_weights = [estimate_font_weight(c['fontname']) for c in sentence_chars]
                avg_font_weight = sum(font_weights) / len(font_weights) if font_weights else 400

                bold = avg_font_weight >= 700
                caps_ratio = calculate_caps_ratio(sentence)
                num_words = len(sentence.split())
                text_len = len(sentence)
                digit_ratio = sum(1 for ch in sentence if ch.isdigit()) / text_len if text_len else 0
                starts_with_number = sentence[0].isdigit() if sentence else False
                contains_colon = ':' in sentence
                indentation = min(w['x0'] for w in sentence_words) if sentence_words else 0
                line_height = avg_font_size * 1.5

                writer.writerow({
                    'page': page_num,
                    'sentence_index': sentence_idx,
                    'text': sentence,
                    'avg_font_size': round(avg_font_size, 2),
                    'bold': bold,
                    'caps_ratio': round(caps_ratio, 2),
                    'indentation': round(indentation, 2),
                    'num_words': num_words,
                    'text_len': text_len,
                    'digit_ratio': round(digit_ratio, 2),
                    'starts_with_number': starts_with_number,
                    'contains_colon': contains_colon,
                    'line_height': round(line_height, 2),
                })

    print(f"Features extracted and saved to: {csv_path}")

def main():
    pdf_path = r"D:\Adobe\Pdfs\TOPJUMP-PARTY-INVITATION-20161003-V01.pdf"  # Replace with your actual PDF path
    csv_path = "extracted_features_sentences5.csv"
    extract_features_to_csv(pdf_path, csv_path)

if __name__ == "__main__":
    main()
