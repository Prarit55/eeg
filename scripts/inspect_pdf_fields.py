import fitz

pdf_path = r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\.user_uploaded\media_1795529408158_d320ee9a.pdf"
doc = fitz.open(pdf_path)

targets = [
    (30, "[framework and version]"),
    (34, "[enter the actual number"),
    (36, "[enter values]"),
    (36, "[enter value]"),
    (39, "[Note for submission"),
    (45, "[optionally add"),
    (48, "[add milestone dates")
]

for page_idx, search_text in targets:
    page = doc[page_idx]
    rects = page.search_for(search_text)
    print(f"=== Page {page_idx + 1} for '{search_text}' ===")
    print("Found rects:", rects)
    drawings = page.get_drawings()
    yellow_drawings = []
    for d in drawings:
        fill = d.get("fill")
        if fill and fill[0] > 0.8 and fill[1] > 0.8 and fill[2] < 0.4:
            yellow_drawings.append((d["rect"], fill))
    print(f"Yellow highlight rects on page {page_idx + 1}:", yellow_drawings)
