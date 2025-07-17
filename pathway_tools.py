import pathway as pw
import os
import tempfile
import fitz  # This is PyMuPDF
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # Light and fast

@pw.udf
def extract_pdf_text(binary: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(binary)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    os.remove(tmp_path)

    return text

SECTION_HEADERS = {
    "abstract", "introduction", "related work", "method", "methods",
    "methodology", "conclusion", "results", "discussion", "references", "background"
}

@pw.udf
def clean_and_split(text: str) -> list[str]:
    # Lowercase for easier matching
    text = text.lower()

    # Split by newlines to isolate section headers
    lines = text.split('\n')
    cleaned_lines = []

    skip_next_line_if_header = False
    for line in lines:
        line = line.strip()

        # Skip known section headers
        if any(header in line and len(line.split()) <= 4 for header in SECTION_HEADERS):
            skip_next_line_if_header = True
            continue

        # Optionally skip the line immediately after a header (if it’s just spacing or junk)
        if skip_next_line_if_header and len(line) < 10:
            skip_next_line_if_header = False
            continue

        skip_next_line_if_header = False
        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)

    # Remove numbers, decimals, percents
    text = re.sub(r"\b\d+(\.\d+)?%?\b", "", text)

    # Remove inline equations like "n = 5"
    text = re.sub(r"\b\w+\s*=\s*[^,;\.\n]+", "", text)

    # Remove short math expressions like "3 * x"
    text = re.sub(r"\b\d+\s*[*\/^+-]\s*\w+", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Split at punctuation boundaries
    parts = re.split(r"[,;\.]", text)

    # Strip and remove short fragments
    return [p.strip() for p in parts if len(p.strip()) > 5]

@pw.udf
def embed_sentences(sentences: list[str]) -> list[list[float]]:
    return [model.encode(s, normalize_embeddings=True).tolist() for s in sentences]