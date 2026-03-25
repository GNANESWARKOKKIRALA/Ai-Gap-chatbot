"""
loader.py — Parse PDF, TXT, DOCX, CSV into plain text
"""
import os
import pdfplumber
import docx
import pandas as pd


def load_file(filepath: str) -> str:
    """Load any supported file and return its text content."""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return _load_pdf(filepath)
    elif ext in (".txt", ".md"):
        return _load_txt(filepath)
    elif ext == ".docx":
        return _load_docx(filepath)
    elif ext == ".csv":
        return _load_csv(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf(path: str) -> str:
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text.append(t)
    return "\n\n".join(text)


def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_csv(path: str) -> str:
    df = pd.read_csv(path)
    return df.to_string(index=False)
