from PyPDF2 import PdfReader

r = PdfReader(r'c:\Users\brr33\Downloads\project Ather.pdf')
for i, page in enumerate(r.pages):
    print(f"--- PAGE {i+1} ---")
    print(page.extract_text())
