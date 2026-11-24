import fitz

doc = fitz.open(r"D:\Desktop\Parth\CODES\eeg\eeg\Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages.pdf")

for p_num in [1, 2, 3, 4]:
    page = doc[p_num - 1]
    blocks = page.get_text("dict")["blocks"]
    print(f"\n================ PAGE {p_num} ================")
    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                line_text = "".join([s["text"] for s in l["spans"]]).strip()
                if any(w in line_text for w in ["Parth", "Saini", "06219051923", "Name:", "Enrollment"]):
                    for s in l["spans"]:
                        print(f"text='{s['text']}' | font={s['font']} | size={s['size']:.2f} | flags={s['flags']} | bbox={s['bbox']}")
