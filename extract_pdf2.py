import fitz  # PyMuPDF

doc = fitz.open(r'c:\Users\brr33\Downloads\project Ather.pdf')
for i, page in enumerate(doc):
    print(f"--- PAGE {i+1} ---")
    text = page.get_text()
    if text.strip():
        print(text)
    else:
        print("[No text extracted - may be image-based]")
doc.close()
