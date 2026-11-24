import fitz

pdf_path = r"D:\Desktop\Parth\CODES\eeg\eeg\Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages.pdf"
doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")

for i, page in enumerate(doc):
    text = page.get_text("text")
    for term in ["Parth", "Saini", "06219051923"]:
        if term.lower() in text.lower():
            rects = page.search_for(term)
            print(f"Page {i+1} has '{term}' -> {len(rects)} rects:")
            for r in rects:
                print(f"   {r}")
