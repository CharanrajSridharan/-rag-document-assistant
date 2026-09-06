import pdfplumber

def extract_all_pages(pdf_path: str) -> str:
    all_pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_pages_text.append(text)
    full_text = "\n".join(all_pages_text)
    return full_text

if __name__ == "__main__":
    result = extract_all_pages("data/sample.pdf")
    print(f"Total characters extracted: {len(result)}")
    print(result[:1000])  # just print the first 1000 characters as a preview