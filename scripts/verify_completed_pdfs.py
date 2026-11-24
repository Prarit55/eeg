import fitz

def verify_pdf(pdf_path, name):
    doc = fitz.open(pdf_path)
    print(f"\n==================================================")
    print(f"VERIFYING {name}: {pdf_path}")
    print(f"Total Pages: {len(doc)}")
    
    yellow_found = []
    placeholders_found = []
    
    for i, page in enumerate(doc):
        # Check text
        text = page.get_text("text")
        for bad in ["[enter", "[add milestone", "[optionally", "[framework", "[Note for submission"]:
            if bad.lower() in text.lower():
                placeholders_found.append((i + 1, bad))
        
        # Check yellow drawings
        for d in page.get_drawings():
            fill = d.get("fill")
            if fill and fill[0] > 0.8 and fill[1] > 0.8 and fill[2] < 0.4:
                yellow_found.append((i + 1, d["rect"]))
                
    print(f"Remaining placeholders: {placeholders_found}")
    print(f"Remaining yellow highlights: {len(yellow_found)} found -> {yellow_found}")
    
    # Render key modified pages
    pages_to_render = [31, 35, 37, 40, 46, 49, 50]
    out_dir = fitz.open() # dummy
    for p_num in pages_to_render:
        if p_num <= len(doc):
            pix = doc[p_num - 1].get_pixmap(dpi=150)
            png_name = f"C:\\Users\\Parth\\.gemini\\antigravity\\brain\\d2e435ee-fa6a-4133-912b-906825629c9d\\scratch\\{name}_p{p_num}.png"
            pix.save(png_name)
    print(f"Rendered sample pages to scratch/{name}_p*.png")

verify_pdf(r"D:\Desktop\Parth\CODES\eeg\eeg\Minor_Project_Report_Query_Adaptive_Self_RAG_Completed.pdf", "version_A")
verify_pdf(r"D:\Desktop\Parth\CODES\eeg\eeg\Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages.pdf", "version_B")
