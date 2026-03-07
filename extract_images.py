import fitz  # PyMuPDF
from PIL import Image
import io

doc = fitz.open(r'c:\Users\brr33\Downloads\project Ather.pdf')

# Convert pages to images and save them
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
    img_path = f"page_{i+1}.png"
    pix.save(img_path)
    print(f"Saved {img_path}")
    
doc.close()
print("Done! Now need Tesseract OCR to extract text from images.")
