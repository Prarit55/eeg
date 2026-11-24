import fitz
from pathlib import Path
import shutil
import tempfile

def apply_student_info(doc):
    font_reg_path = r"C:\Windows\Fonts\calibri.ttf"
    font_bold_path = r"C:\Windows\Fonts\calibrib.ttf"
    font_times_bd_path = r"C:\Windows\Fonts\timesbd.ttf"

    font_bold = fitz.Font(fontfile=font_bold_path)
    font_times_bd = fitz.Font(fontfile=font_times_bd_path)

    # -------------------------------------------------------------------------
    # PAGE 1 (Cover Page)
    # -------------------------------------------------------------------------
    p1 = doc[0]
    r1_name = fitz.Rect(180.0, 485.0, 480.0, 506.0)
    r1_enr = fitz.Rect(180.0, 516.0, 480.0, 537.0)
    p1.add_redact_annot(r1_name, fill=(1, 1, 1))
    p1.add_redact_annot(r1_enr, fill=(1, 1, 1))
    p1.apply_redactions()

    t_name = "Name: Prarit Arora"
    w_name = font_times_bd.text_length(t_name, fontsize=14.0)
    x_name = (p1.rect.width - w_name) / 2
    p1.insert_text(fitz.Point(x_name, 499.5), t_name, fontfile=font_times_bd_path, fontsize=14.0, color=(0, 0, 0))

    t_enr = "Enrollment Number: 04919051623"
    w_enr = font_times_bd.text_length(t_enr, fontsize=14.0)
    x_enr = (p1.rect.width - w_enr) / 2
    p1.insert_text(fitz.Point(x_enr, 530.5), t_enr, fontfile=font_times_bd_path, fontsize=14.0, color=(0, 0, 0))

    # -------------------------------------------------------------------------
    # PAGE 2 (Certificate by Supervisor)
    # -------------------------------------------------------------------------
    p2 = doc[1]
    r2_line = fitz.Rect(316.0, 133.0, 375.0, 150.0)
    p2.add_redact_annot(r2_line, fill=(1, 1, 1))
    p2.apply_redactions()
    p2.insert_text(fitz.Point(318.70, 145.2), "Prarit Arora", fontfile=font_bold_path, fontsize=12.0, color=(0, 0, 0))

    # -------------------------------------------------------------------------
    # PAGE 3 (Declaration by Student)
    # -------------------------------------------------------------------------
    p3 = doc[2]
    r3_p1 = fitz.Rect(107.0, 101.0, 166.0, 117.0)
    p3.add_redact_annot(r3_p1, fill=(1, 1, 1))
    r3_sig = fitz.Rect(98.0, 322.0, 260.0, 357.0)
    p3.add_redact_annot(r3_sig, fill=(1, 1, 1))
    p3.apply_redactions()

    p3.insert_text(fitz.Point(109.22, 112.8), "Prarit Arora", fontfile=font_bold_path, fontsize=12.0, color=(0, 0, 0))
    p3.insert_text(fitz.Point(100.02, 334.0), "Prarit Arora", fontfile=font_bold_path, fontsize=12.0, color=(0, 0, 0))
    p3.insert_text(fitz.Point(100.02, 352.0), "Enrollment No. 04919051623", fontfile=font_reg_path, fontsize=12.0, color=(0, 0, 0))

    # -------------------------------------------------------------------------
    # PAGE 4 (Acknowledgement)
    # -------------------------------------------------------------------------
    p4 = doc[3]
    r4_sig = fitz.Rect(495.0, 345.0, 570.0, 363.0)
    p4.add_redact_annot(r4_sig, fill=(1, 1, 1))
    p4.apply_redactions()

    w_p4 = font_bold.text_length("Prarit Arora", fontsize=12.0)
    p4.insert_text(fitz.Point(566.0 - w_p4, 358.0), "Prarit Arora", fontfile=font_bold_path, fontsize=12.0, color=(0, 0, 0))


def process_file(in_path, out_path):
    doc = fitz.open(in_path)
    apply_student_info(doc)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        temp_out = tf.name
    doc.save(temp_out)
    doc.close()
    shutil.copyfile(temp_out, out_path)
    Path(temp_out).unlink(missing_ok=True)
    print(f"[Done] Updated {in_path.name} -> {out_path.name}")


if __name__ == "__main__":
    eeg_dir = Path(r"D:\Desktop\Parth\CODES\eeg\eeg")
    root_dir = Path(r"D:\Desktop\Parth\CODES\eeg")
    docs_dir = eeg_dir / "docs"

    # Version B (54 Pages)
    vB_in = eeg_dir / "Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages.pdf"
    vB_out = eeg_dir / "Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_54Pages_Prarit_Arora.pdf"
    process_file(vB_in, vB_out)
    process_file(vB_in, vB_in)

    shutil.copyfile(vB_out, docs_dir / vB_out.name)
    shutil.copyfile(vB_out, root_dir / vB_out.name)
    shutil.copyfile(vB_in, docs_dir / vB_in.name)
    shutil.copyfile(vB_in, root_dir / vB_in.name)

    # Version A (with dedicated Gantt chart page, 55 pages)
    vA_in = eeg_dir / "Minor_Project_Report_Query_Adaptive_Self_RAG_Completed.pdf"
    vA_out = eeg_dir / "Minor_Project_Report_Query_Adaptive_Self_RAG_Completed_Prarit_Arora.pdf"
    process_file(vA_in, vA_out)
    process_file(vA_in, vA_in)

    shutil.copyfile(vA_out, docs_dir / vA_out.name)
    shutil.copyfile(vA_out, root_dir / vA_out.name)
    shutil.copyfile(vA_in, docs_dir / vA_in.name)
    shutil.copyfile(vA_in, root_dir / vA_in.name)

    print("All files processed and mirrored successfully!")
