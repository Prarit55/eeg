import fitz

pdf_path = r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\.user_uploaded\media_1795529408158_d320ee9a.pdf"
doc = fitz.open(pdf_path)

def inspect_page_spans(page_num, target_word):
    page = doc[page_num - 1]
    blocks = page.get_text("dict")["blocks"]
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    if target_word.lower() in s["text"].lower():
                        print(f"Page {page_num}: text='{s['text']}' | font={s['font']} | size={s['size']:.2f} | color={s['color']} | bbox={s['bbox']}")

inspect_page_spans(31, "Deep learning")
inspect_page_spans(31, "framework")
inspect_page_spans(35, "enter the actual")
inspect_page_spans(37, "Optimiser")
inspect_page_spans(37, "enter values")
inspect_page_spans(40, "Note for submission")
inspect_page_spans(46, "optionally")
inspect_page_spans(49, "add milestone")
inspect_page_spans(49, "All seven objectives")
