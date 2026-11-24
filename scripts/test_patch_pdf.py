import fitz

src_pdf = r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\.user_uploaded\media_1795529408158_d320ee9a.pdf"
doc = fitz.open(src_pdf)

print(f"Loaded PDF with {len(doc)} pages.")

# Test Page 31 (0-indexed: 30)
p31 = doc[30]
# Yellow rect on page 31: Rect(191.5, 471.30, 290.85, 482.85)
# Let's add a redaction annotation over this rect
r31 = fitz.Rect(191.0, 469.0, 291.5, 484.0)
p31.add_redact_annot(r31, fill=(1, 1, 1))
p31.apply_redactions()
# Insert replacement text
p31.insert_text(fitz.Point(193.0, 480.0), "PyTorch 2.6.0", fontname="times-roman", fontsize=10.0, color=(0, 0, 0))

# Test Page 35 (0-indexed: 34)
p35 = doc[34]
# Yellow rects:
# Rect(99.2, 403.5, 559.85, 416.05)
# Rect(99.2, 422.4, 208.25, 435.05)
r35 = fitz.Rect(98.0, 402.0, 562.0, 436.0)
p35.add_redact_annot(r35, fill=(1, 1, 1))
p35.apply_redactions()
txt35 = (
    "The network comprises 3 Conv1D layers (Layer 1: F=64, J=7, stride 1; Layer 2: F=128, J=5, stride 1; "
    "Layer 3: F=256, J=3, stride 1), each followed by BatchNorm1d, ReLU, and MaxPool1d (stride 2). The "
    "BiLSTM has H=128 hidden units per direction (2H=256), followed by temporal global average pooling, "
    "a linear projection bottleneck (d=128), LayerNorm, and Dropout (p=0.30)."
)
p35.insert_textbox(fitz.Rect(99.3, 403.0, 563.0, 436.0), txt35, fontname="times-roman", fontsize=9.2, color=(0, 0, 0))

# Save test output
test_out = r"C:\Users\Parth\.gemini\antigravity\brain\d2e435ee-fa6a-4133-912b-906825629c9d\scratch\test_patched.pdf"
doc.save(test_out)
print(f"Saved test patched PDF to {test_out}")
