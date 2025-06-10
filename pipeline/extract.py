from PyPDF2 import PdfReader

def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]
