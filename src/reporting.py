from __future__ import annotations

from html import escape
from pathlib import Path


def create_forensic_report(
    report_path: Path,
    original_path: Path,
    mask_path: Path,
    annotated_path: Path,
    summary: dict[str, object],
    metadata: dict[str, str],
) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as error:
        raise RuntimeError("PDF reports require the reportlab package") from error

    report_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(report_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#173d3a"))
    story = [Paragraph("Image Forensic Analysis Report", title_style), Spacer(1, 5 * mm)]
    story.append(Paragraph("Automated screening output. This report is not a certified forensic opinion or proof of image authenticity.", styles["BodyText"]))
    story.append(Spacer(1, 4 * mm))

    area_percent = float(summary.get("area_percent", 0.0))
    confidence_percent = float(summary.get("confidence_percent", 0.0))
    details = [
        ["Verdict", escape(str(summary.get("status", "Unknown")))],
        ["Method", escape(str(summary.get("method", "Unknown")))],
        ["Forgery type", escape(str(summary.get("forgery_type", "Unknown")))],
        ["Confidence", f"{confidence_percent:.1f}%"],
        ["Detected area", f"{area_percent:.1f}%"],
        ["Bounding boxes", str(summary.get("box_count", 0))],
        ["Image dimensions", escape(str(summary.get("dimensions", "Unknown")))],
    ]
    table = Table(details, colWidths=[42 * mm, 125 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf1ef")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#18312f")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad6d3")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 5 * mm), Paragraph("Image evidence", styles["Heading2"])])

    evidence = []
    for label, path in (("Original", original_path), ("Detected mask", mask_path), ("Bounding boxes", annotated_path)):
        image = Image(str(path), width=54 * mm, height=44 * mm, kind="proportional")
        evidence.append([Paragraph(label, styles["BodyText"]), image])
    evidence_table = Table([evidence], colWidths=[56 * mm] * 3)
    evidence_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(evidence_table)
    story.extend([Spacer(1, 5 * mm), Paragraph("Embedded metadata", styles["Heading2"])])
    metadata_rows = [[escape(key), escape(value)] for key, value in metadata.items()] or [["Metadata", "No readable EXIF metadata found"]]
    metadata_table = Table(metadata_rows, colWidths=[52 * mm, 115 * mm])
    metadata_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cad6d3")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f7f6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("The system analyzes visible image patterns and embedded metadata. Metadata can be removed or altered, and model results depend on training data and image provenance.", styles["BodyText"]))
    document.build(story)