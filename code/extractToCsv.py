import pdfplumber
import csv

def estimate_font_weight(fontname: str) -> int:
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

def extract_features_to_csv(pdf_path, csv_path):
    with pdfplumber.open(pdf_path) as pdf, open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['page', 'word_index', 'text', 'font_size', 'font_weight', 
                      'x0', 'top', 'indentation', 'word_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            for idx, word_info in enumerate(words):
                text = word_info.get('text', '')
                x0 = word_info.get('x0', 0)
                x1 = word_info.get('x1', 0)
                top = word_info.get('top', 0)
                bottom = word_info.get('bottom', 0)

                word_chars = [
                    char for char in page.chars
                    if (char['x0'] >= x0 and char['x1'] <= x1 and
                        char['top'] >= top and char['bottom'] <= bottom)
                ]

                if word_chars:
                    avg_font_size = sum(char['size'] for char in word_chars) / len(word_chars)
                    avg_font_weight = sum(estimate_font_weight(char['fontname']) for char in word_chars) / len(word_chars)
                else:
                    avg_font_size = 0
                    avg_font_weight = 400

                indentation = x0
                word_length = len(text)

                writer.writerow({
                    'page': page_num,
                    'word_index': idx,
                    'text': text,
                    'font_size': round(avg_font_size, 2),
                    'font_weight': int(round(avg_font_weight)),
                    'x0': round(x0, 2),
                    'top': round(top, 2),
                    'indentation': round(indentation, 2),
                    'word_length': word_length
                })

    print(f"Features extracted and saved to: {csv_path}")

def main():
    pdf_path = "D:\Adobe\Pdfs\E0CCG5S312.pdf"       # Replace this with your PDF file path
    csv_path = "extracted_features.csv"
    extract_features_to_csv(pdf_path, csv_path)

if __name__ == "__main__":
    main()
