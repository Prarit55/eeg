import fitz

doc = fitz.open(r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\.user_uploaded\media_1795529408158_d320ee9a.pdf")
p37 = doc[36]

blocks = p37.get_text("dict")["blocks"]
for b in blocks:
    if "lines" in b:
        for l in b["lines"]:
            for s in l["spans"]:
                if any(w in s["text"] for w in ["256 Hz", "0.5–60", "50 / 60", "Binary cross", "enter", "384", "2,571"]):
                    print(f"{s['text']} | bbox={s['bbox']}")
