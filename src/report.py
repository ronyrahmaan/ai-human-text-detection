"""Build a downloadable analysis report (PDF or plain text)."""
from __future__ import annotations

from datetime import datetime


def _ascii(text: str) -> str:
    """fpdf2 core fonts are latin-1; drop anything they cannot render."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _summary_lines(text: str, selected: str, prediction: dict,
                   all_probs: dict[str, float], stats: dict[str, float]) -> list[str]:
    lines = [
        "AI vs. Human Text Detection - Analysis Report",
        datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
        "",
        f"Selected model: {selected}",
        f"Verdict: {prediction['label']}  (confidence {prediction['confidence']:.1%})",
        "",
        "All models, probability the text is AI-generated:",
    ]
    for name, p in sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {name:<16} {p:.1%}  -> {'AI' if p >= 0.5 else 'Human'}")

    lines += ["", "Text statistics:"]
    for key in ("word_count", "avg_sentence_length", "sentence_length_std",
                "type_token_ratio", "flesch_reading_ease"):
        if key in stats:
            lines.append(f"  {key.replace('_', ' '):<22} {stats[key]:.2f}")

    preview = " ".join(text.split())[:600]
    lines += ["", "Analyzed text (first 600 characters):", preview,
              "",
              "Note: automated AI-text detection is probabilistic and can be wrong, "
              "especially on paraphrased text or writing by non-native English "
              "speakers. Treat this as a signal, not proof."]
    return lines


def text_report(text, selected, prediction, all_probs, stats) -> bytes:
    lines = _summary_lines(text, selected, prediction, all_probs, stats)
    return ("\n".join(lines)).encode("utf-8")


def pdf_report(text, selected, prediction, all_probs, stats) -> bytes:
    from fpdf import FPDF

    lines = _summary_lines(text, selected, prediction, all_probs, stats)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 10, _ascii(lines[0]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for line in lines[1:]:
        if line.strip():
            pdf.multi_cell(pdf.epw, 6, _ascii(line))
        else:
            pdf.ln(4)
    return bytes(pdf.output())
