# Adobe Hackathon: Document Outline Extractor (Challenge 1A)

## Overview

This project is a solution for **Challenge 1A** of the Adobe India Hackathon. It is a Python-based system that uses a pre-trained machine learning model to analyze the structure of a PDF document and extract a hierarchical outline, including the document title and headings (H1, H2, H3). The solution is designed to be **fast, accurate, and fully compliant** with the hackathon's offline and resource constraints.

---

## Project Structure

The project is organized for containerized execution using Docker.

```
ADOBE_1A/
├── input/
│   ├── document1.pdf
│   └── ...
├── output/
│   ├── document1.json
│   └── ...
├── Dockerfile
├── README.md
├── requirements.txt
├── solution.py
└── trained_model.joblib  
```

---

## Approach

The solution uses a **machine learning-based pipeline** to move beyond simple rule-based heuristics and achieve a higher level of accuracy in identifying document structure.

### 1. Detailed Feature Extraction

The process begins by using the **PyMuPDF** library to parse the input PDF and extract every text segment along with a rich set of layout and style features. For each segment, it calculates **13 distinct features**, including:

- **Font Properties:** `font_size`, `font_weight`
- **Positional Data:** `x0`, `top`, `indentation`, `relative_x0`, `relative_top`, `line_spacing_above`
- **Textual Content:** `word_length`, `char_length`, `is_all_caps`, `has_bullet_or_number`
- **Document Context:** `page number`

### 2. Machine Learning Classification

The core of the solution is a **pre-trained RandomForestClassifier** loaded from the `trained_model.joblib` file. This model takes the 13 features for each text block as input and predicts its structural role (e.g., title, H1, H2, H3, or other). This data-driven approach allows the system to accurately identify headings even in documents with complex or unconventional formatting.

### 3. Hierarchical Post-Processing

After the initial classification, the script performs several post-processing steps to refine the outline:

- **Title Identification:** Uses a heuristic to identify the most likely candidate for the main document title on the first page and corrects its label.
- **Multi-Line Title Assembly:** Intelligently groups consecutive lines that share the same style as the identified title to form a complete, multi-line title.
- **Hierarchy Correction:** Ensures a logical heading structure (e.g., an H3 cannot follow an H1 without an intermediate H2) by adjusting the levels of any out-of-place headings.

### 4. JSON Output Generation

Finally, the script constructs the required JSON output, containing the document title and a clean, ordered list of all identified headings with their level, text, and page number.

---

## Open Source Libraries Used

- **PyMuPDF (fitz):** For robust and efficient PDF parsing and text extraction.
- **Pandas:** For organizing the extracted features into a structured DataFrame for the machine learning model.
- **Scikit-learn (joblib):** For loading and using the pre-trained RandomForestClassifier model.

---

## How to Build and Run

The solution is containerized using Docker for easy and consistent execution. All commands should be run from the root directory of the project (e.g., `ADOBE_1A/`).

### Prerequisites

- Docker installed on your machine.

### 1. Build the Docker Image

Navigate to the root of your project directory and run the following command. This will build the Docker image, install all dependencies, and copy your script and the pre-trained model into the image.

```bash
docker build --platform linux/amd64 -t outline-extractor .
```

### 2. Run the Solution

Once the image is built, you can run it on a directory of PDFs. The command mounts your local input directory into the container and mounts an output directory to store the results. The `--network none` flag ensures the container runs in a completely offline environment.

```bash
docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none outline-extractor
```

The script will automatically process all PDF files found in the input directory and generate a corresponding `.json` file for each one in the output directory.

---

## How to Test with New PDFs

To test the solution with new documents, simply:

1. Place any new PDF files you want to analyze into the `input` folder.
2. Run the `docker run` command as shown above.
3. Check the `output` folder for the newly generated JSON files.

---
