# This module handles reading different file types and turning them into raw text

import PyPDF2
import docx2txt
import io

def load_document(uploaded_file) -> str:
    """
    Accepts a Streamlit UploadedFile object.
    Detects its type and extracts all text from it.
    Returns a single string of the full document text.
    """
    file_name = uploaded_file.name.lower()
    
    # Read the raw bytes from the uploaded file
    file_bytes = uploaded_file.read()
    
    if file_name.endswith(".pdf"):
        return _extract_pdf_text(file_bytes)
    
    elif file_name.endswith(".txt"):
        # Simple decode for plain text
        return file_bytes.decode("utf-8", errors="ignore")
    
    elif file_name.endswith(".docx"):
        # docx2txt needs a file-like object
        return docx2txt.process(io.BytesIO(file_bytes))
    
    else:
        raise ValueError(f"Unsupported file type: {file_name}")


def _extract_pdf_text(file_bytes: bytes) -> str:
    """
    Uses PyPDF2 to iterate over all pages and extract text.
    Some PDFs are scanned images — they'll return empty strings (known limitation).
    """
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page_text
    
    return text