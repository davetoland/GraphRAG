# ingestion/reader.py
from unstructured.partition.auto import partition
from pathlib import Path

def read_file(path: Path):
    # auto-detects PDF, DOCX, HTML, EML, etc.
    elements = partition(str(path))
    # return list of non-empty text blocks
    return [el.text for el in elements if el.text and el.text.strip()]
