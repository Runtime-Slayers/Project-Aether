import easyocr
import os

reader = easyocr.Reader(['en'], gpu=False)

output_text = []
for i in range(1, 11):
    img_path = f"page_{i}.png"
    if os.path.exists(img_path):
        print(f"Processing {img_path}...")
        result = reader.readtext(img_path, detail=0, paragraph=True)
        page_text = "\n".join(result)
        output_text.append(f"=== PAGE {i} ===\n{page_text}\n")
        print(f"Page {i} done.")

# Save full extracted text
with open("project_content.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_text))

print("\nFull content saved to project_content.txt")
print("\n" + "="*50)
print("EXTRACTED CONTENT:")
print("="*50)
print("\n".join(output_text))
