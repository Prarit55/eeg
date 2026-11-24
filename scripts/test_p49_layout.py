import fitz

src_pdf = r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\.user_uploaded\media_1795529408158_d320ee9a.pdf"
doc = fitz.open(src_pdf)
page = doc[48] # Page 49 (0-indexed)

# Let's inspect the yellow rect on page 49:
# Yellow highlight rect: Rect(120.2, 675.2, 522.05, 688.95)
# And the sentence ends at y = 689.0
print("Page 49 height:", page.rect.height)
print("Page 49 width:", page.rect.width)

# What if we redact the yellow box and insert milestone text,
# AND check if a compact Gantt table fits or if a dedicated page is cleaner?
