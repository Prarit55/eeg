import fitz

doc = fitz.open(r"D:\Desktop\Parth\CODES\eeg\eeg\Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages.pdf")

# Let's inspect available system fonts or built-in fonts in fitz
print("Page 1 fonts:", doc[0].get_fonts())
print("Page 2 fonts:", doc[1].get_fonts())
print("Page 3 fonts:", doc[2].get_fonts())
print("Page 4 fonts:", doc[3].get_fonts())
