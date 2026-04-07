"""
Document Loader - Reads PDF files and extracts text page by page using PyMuPDF.
"""

import os
import fitz  # PyMuPDF


def load_pdfs(folder_path: str) -> list[dict]:
    """
    Reads every PDF file from the given folder and extracts text from each page.

    Args:
        folder_path: Path to the directory containing PDF files.

    Returns:
        A list of dictionaries, each with:
            - "text": the extracted page text
            - "source": the PDF filename
            - "page": the page number (1-indexed)
    """
    documents = []

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return documents

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
            pdf = fitz.open(filepath)

            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()

                documents.append({
                    "text": text,
                    "source": filename,
                    "page": page_num + 1,  # 1-indexed page number
                })

            num_pages = len(pdf)
            pdf.close()
            print(f"Loaded: {filename} ({num_pages} pages)")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return documents


if __name__ == "__main__":
    raw_folder = os.path.join(".", "data", "raw")
    pages = load_pdfs(raw_folder)
    print(f"\nTotal pages loaded: {len(pages)}")
