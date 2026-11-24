import fitz

font_reg = fitz.Font(fontfile=r"C:\Windows\Fonts\calibri.ttf")
font_bold = fitz.Font(fontfile=r"C:\Windows\Fonts\calibrib.ttf")
font_times_bd = fitz.Font(fontfile=r"C:\Windows\Fonts\timesbd.ttf")

print("--- Page 1 ---")
w_name = font_times_bd.text_length("Name: Prarit Arora", fontsize=14.0)
w_enr = font_times_bd.text_length("Enrollment Number: 04919051623", fontsize=14.0)
print(f"Name tw={w_name:.2f}, x={(595.4 - w_name)/2:.2f}")
print(f"Enrollment tw={w_enr:.2f}, x={(595.4 - w_enr)/2:.2f}")

print("\n--- Page 2 Line 3 ---")
w_pre = font_reg.text_length("bonafide record of the work carried out by ", fontsize=12.0)
w_name_p2 = font_bold.text_length("Prarit Arora", fontsize=12.0)
w_post = font_reg.text_length(", a student of Bachelor of Technology", fontsize=12.0)
total_p2 = w_pre + w_name_p2 + w_post
print(f"pre={w_pre:.2f}, name={w_name_p2:.2f}, post={w_post:.2f}, total={total_p2:.2f} (max allowed: {563.0 - 100.43:.2f})")

print("\n--- Page 3 Line 1 ---")
w_i = font_reg.text_length("I, ", fontsize=12.0)
w_name_p3 = font_bold.text_length("Prarit Arora", fontsize=12.0)
w_post_p3 = font_reg.text_length(", a student of Bachelor of Technology (B.Tech), 7th Semester, University School of", fontsize=12.0)
total_p3 = w_i + w_name_p3 + w_post_p3
print(f"I={w_i:.2f}, name={w_name_p3:.2f}, post={w_post_p3:.2f}, total={total_p3:.2f} (max allowed: {563.0 - 100.02:.2f})")

print("\n--- Page 4 ---")
w_name_p4 = font_bold.text_length("Prarit Arora", fontsize=12.0)
print(f"name={w_name_p4:.2f}, x={566.0 - w_name_p4:.2f}")
