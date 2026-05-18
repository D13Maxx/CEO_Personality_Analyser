"""PDF report generator for personality profiles."""

import os
import tempfile
from datetime import datetime

from fpdf import FPDF
import plotly.graph_objects as go

from src.visualization.percentile_bands import get_band_category
from config import TRAITS, TRAIT_LABELS, REPORT_TITLE


class PDFReport(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, REPORT_TITLE, 0, 1, "C")
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Automated Linguistic Big Five Analysis", 0, 1, "C")
        self.ln(10)

    def footer(self):
        self.set_y(-25)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.multi_cell(0, 4, (
            "Disclaimer: This profile is generated via an automated ML model analyzing "
            "linguistic patterns (Harrison et al., 2019). It represents stylistic tendencies "
            "in public speech, not a clinical psychological assessment."
        ), 0, "C")
        self.set_y(-10)
        self.cell(0, 4, f"Page {self.page_no()}", 0, 0, "C")


def generate_pdf_report(ceo_name, company_name, word_count, scores, percentiles, radar_fig):
    pdf = PDFReport()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, f"Subject: {ceo_name}", 0, 1, "L")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    if company_name and company_name != "Unknown":
        pdf.cell(0, 6, f"Company/Context: {company_name}", 0, 1, "L")

    pdf.cell(0, 6, f"Analysis Date: {datetime.now():%B %d, %Y} | "
             f"Word Count: {word_count:,}", 0, 1, "L")
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(10)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        chart_path = tmp.name

    try:
        radar_fig.update_layout(width=600, height=500)
        radar_fig.write_image(chart_path, scale=2)
        pdf.image(chart_path, x=(210 - 120) / 2, y=pdf.get_y(), w=120)
        pdf.set_y(pdf.get_y() + 105)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"[Chart failed: {e}]", 0, 1, "C")
        pdf.ln(10)
    finally:
        if os.path.exists(chart_path):
            os.remove(chart_path)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Trait Breakdown", 0, 1, "L")

    pdf.set_fill_color(241, 245, 249)
    pdf.set_font("Helvetica", "B", 10)
    col_widths = [60, 30, 40, 50]
    for label, w in zip(["Personality Trait", "Score (1-7)", "Percentile", "Category"], col_widths):
        pdf.cell(w, 8, label, 1, 0, "C", True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for trait in TRAITS:
        score = scores.get(trait, 0.0)
        pct = percentiles.get(trait, 0.0)
        band, _ = get_band_category(pct)

        pdf.cell(col_widths[0], 8, TRAIT_LABELS[trait], 1, 0, "L")
        pdf.cell(col_widths[1], 8, f"{score:.2f}", 1, 0, "C")
        pdf.cell(col_widths[2], 8, f"{int(round(pct))}th", 1, 0, "C")

        band_colors = {"High": (16, 185, 129), "Low": (239, 68, 68)}
        pdf.set_text_color(*band_colors.get(band, (245, 158, 11)))
        pdf.cell(col_widths[3], 8, band, 1, 1, "C")
        pdf.set_text_color(15, 23, 42)

    return bytes(pdf.output())
