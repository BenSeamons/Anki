# pdf_to_los.py
import argparse
import re
import csv
from pathlib import Path
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def clean_and_split(text: str) -> list[str]:
    """Turn raw PDF text into pseudo-learning objectives."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Try to split on bullet points, numbers, or sentence-like endings
    candidates = re.split(r"(?:•|\d+\)|\d+\.\s|;|\n)", text)

    # Keep only medium-length sentences (not junk or giant paragraphs)
    los = [c.strip() for c in candidates if 15 < len(c.strip()) < 250]

    return los

def save_to_csv(los: list[str], out_path: str):
    """Save extracted objectives to a CSV file with one LO per row."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Objectives"])
        for lo in los:
            writer.writerow([lo])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF lecture into pseudo-LOs (CSV).")
    parser.add_argument("pdf", help="Path to lecture PDF")
    parser.add_argument("--out", default="Objectives.csv",
                        help="Output CSV filename (default: Objectives_fromPDF.csv)")
    args = parser.parse_args()

    text = extract_text_from_pdf(args.pdf)
    los = clean_and_split(text)

    if not los:
        print("⚠️ No LOs could be extracted. Check PDF formatting.")
    else:
        save_to_csv(los, args.out)
        print(f"✅ Extracted {len(los)} objectives → {args.out}")
