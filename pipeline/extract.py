from PyPDF2 import PdfReader

def extract_pages(pdf_path):
    with open(pdf_path, "rb") as pr:
        reader = PdfReader(pr)
        return [page.extract_text() or "" for page in reader.pages]
