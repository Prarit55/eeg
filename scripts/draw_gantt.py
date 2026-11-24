import fitz

def draw_gantt_table(page, rect):
    # rect: fitz.Rect for the table
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    
    # Header: Task | Dates | W1 | W2 | W3 | W4 | W5 | W6 | W7 | W8
    col_w = [140, 68] + [30] * 8
    total_w = sum(col_w)
    scale = width / total_w
    col_w = [w * scale for w in col_w]
    
    tasks = [
        ("Task", "Dates", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"),
        ("Dataset & Requirements", "Sep 14–20", 1),
        ("EEG Preprocessing & Baseline", "Sep 21–27", 2),
        ("Knowledge Graph Construction", "Sep 28–Oct 4", 3),
        ("Vector DB & Retrieval", "Oct 5–11", 4),
        ("Query Decomposition & Scoring", "Oct 12–18", 5),
        ("Self-RAG & Evidence Fusion", "Oct 19–25", 6),
        ("Integration & Evaluation", "Oct 26–Nov 1", 7),
        ("Testing & Documentation", "Nov 2–8", 8),
    ]
    
    row_h = height / len(tasks)
    
    for row_idx, row_data in enumerate(tasks):
        ry0 = y0 + row_idx * row_h
        ry1 = ry0 + row_h
        
        # Header background
        if row_idx == 0:
            page.draw_rect(fitz.Rect(x0, ry0, x1, ry1), color=(0.7, 0.7, 0.7), fill=(0.90, 0.93, 0.96), width=0.5)
        else:
            bg = (1, 1, 1) if row_idx % 2 == 1 else (0.97, 0.98, 0.99)
            page.draw_rect(fitz.Rect(x0, ry0, x1, ry1), color=(0.8, 0.8, 0.8), fill=bg, width=0.5)
            
        cur_x = x0
        for col_idx in range(10):
            cw = col_w[col_idx]
            cell_rect = fitz.Rect(cur_x, ry0, cur_x + cw, ry1)
            page.draw_line(fitz.Point(cur_x, ry0), fitz.Point(cur_x, ry1), color=(0.75, 0.75, 0.75), width=0.5)
            
            if row_idx == 0:
                txt = row_data[col_idx]
                fontname = "helv"
                fontsize = 7.5
                tw = fitz.get_text_length(txt, fontname=fontname, fontsize=fontsize)
                tx = cur_x + (cw - tw) / 2 if col_idx >= 1 else cur_x + 4
                ty = ry0 + row_h * 0.7
                page.insert_text(fitz.Point(tx, ty), txt, fontname=fontname, fontsize=fontsize, color=(0.1, 0.1, 0.2))
            else:
                task_name, dates, active_w = row_data
                if col_idx == 0:
                    page.insert_text(fitz.Point(cur_x + 4, ry0 + row_h * 0.7), task_name, fontname="times-roman", fontsize=7.5, color=(0, 0, 0))
                elif col_idx == 1:
                    tw = fitz.get_text_length(dates, fontname="times-roman", fontsize=7.0)
                    page.insert_text(fitz.Point(cur_x + (cw - tw) / 2, ry0 + row_h * 0.7), dates, fontname="times-roman", fontsize=7.0, color=(0.2, 0.2, 0.2))
                else:
                    week_num = col_idx - 1
                    if week_num == active_w:
                        # Draw black/dark blue progress block
                        bw = cw * 0.65
                        bh = row_h * 0.55
                        bx = cur_x + (cw - bw) / 2
                        by = ry0 + (row_h - bh) / 2
                        page.draw_rect(fitz.Rect(bx, by, bx + bw, by + bh), color=(0.1, 0.2, 0.35), fill=(0.15, 0.25, 0.45))
            cur_x += cw
        page.draw_line(fitz.Point(x1, ry0), fitz.Point(x1, ry1), color=(0.75, 0.75, 0.75), width=0.5)

print("draw_gantt_table defined.")
