
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


def array_to_image(array, residual=False):
    """Convert a NumPy array into an in-memory PNG image."""

    array = np.asarray(array, dtype=np.float32)

    if residual:
        limit = np.percentile(np.abs(array), 99)

        if limit < 1e-8:
            limit = 1.0

        normalized = np.clip(array / limit, -1, 1)

        red = np.clip(normalized, 0, 1)
        blue = np.clip(-normalized, 0, 1)
        green = 1.0 - np.abs(normalized)

        rgb = np.stack([red, green, blue], axis=-1)
        image_array = (rgb * 255).astype(np.uint8)

        image = PILImage.fromarray(image_array, mode="RGB")

    else:
        low = np.percentile(array, 1)
        high = np.percentile(array, 99)

        if high - low < 1e-8:
            high = low + 1.0

        normalized = np.clip(
            (array - low) / (high - low),
            0,
            1
        )

        image_array = (normalized * 255).astype(np.uint8)
        image = PILImage.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def generate_pdf(image_path, report, output_path):
    """Generate a PDF report containing Olin inference results."""

    image_path = Path(image_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Record the report generation time in Indian Standard Time.
    generated_at = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).strftime("%d %B %Y, %I:%M %p IST")

    # Load reconstruction outputs.
    reconstructed_array = np.load(
        report["outputs"]["reconstructed_image"]
    )

    residual_array = np.load(
        report["outputs"]["residual_image"]
    )

    # Prepare the original image for display.
    if image_path.suffix.lower() == ".npy":
        original_array = np.load(image_path).astype(np.float32)
    else:
        with PILImage.open(image_path) as source:
            original_image = source.convert("L")
            original_image = original_image.resize(
                (160, 160),
                PILImage.Resampling.LANCZOS
            )
            original_array = np.asarray(
                original_image,
                dtype=np.float32
            ) / 255.0

    original_buffer = array_to_image(original_array)
    reconstructed_buffer = array_to_image(reconstructed_array)
    residual_buffer = array_to_image(
        residual_array,
        residual=True
    )

    # Create the PDF document.
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "OlinTitle",
        parent=styles["Title"],
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        "OlinSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "OlinSection",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=14,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "OlinNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        spaceAfter=5
    )

    small_style = ParagraphStyle(
        "OlinSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=11
    )

    story = []

    # Title.
    story.append(Paragraph("OLIN", title_style))
    story.append(
        Paragraph(
            "Gravitational Lens Inference Report",
            subtitle_style
        )
    )

    # Analysis information.
    story.append(
        Paragraph("Analysis Information", section_style)
    )

    information_data = [
        ["Input image", escape(str(report["input_image"]))],
        ["Input type", escape(str(report["input_type"]))],
        ["Evaluation type", escape(str(report["evaluation_type"]))],
        ["Model", escape(str(report["model"]))],
        ["MC dropout passes", str(report["mc_passes"])],
        ["Generated on", generated_at],
    ]

    information_table = Table(
        information_data,
        colWidths=[145, 355]
    )

    information_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(information_table)

    # Predicted lens parameters.
    story.append(
        Paragraph("Predicted Lens Parameters", section_style)
    )

    exploratory = report["input_type"] != "synthetic"

    uncertainty_heading = (
        "Synthetic-calibrated uncertainty*"
        if exploratory
        else "Calibrated uncertainty"
    )

    parameter_data = [
        ["Parameter", "Prediction", uncertainty_heading]
    ]

    for name, values in report["lens_parameters"].items():
        parameter_data.append([
            escape(str(name)),
            f'{values["prediction"]:.6f}',
            f'{values["calibrated_uncertainty"]:.6f}'
        ])

    parameter_table = Table(
        parameter_data,
        colWidths=[145, 145, 210],
        repeatRows=1
    )

    parameter_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story.append(parameter_table)

    if exploratory:
        story.append(Spacer(1, 5))
        story.append(
            Paragraph(
                "* Uncertainty calibration was established using "
                "synthetic validation data. It has not been validated "
                "for this input image domain.",
                small_style
            )
        )

    # Fitted source parameters.
    story.append(
        Paragraph("Fitted Source Parameters", section_style)
    )

    source_data = [
        ["Parameter", "Estimated value"]
    ]

    for name, value in report["source_parameters"].items():
        if isinstance(value, (int, float, np.number)):
            source_data.append([
                escape(str(name)),
                f"{float(value):.6f}"
            ])

    source_table = Table(
        source_data,
        colWidths=[250, 250],
        repeatRows=1
    )

    source_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(source_table)

    # Image comparison.
    story.append(
        Paragraph("Image Reconstruction", section_style)
    )

    image_data = [[
        Image(original_buffer, width=145, height=145),
        Image(reconstructed_buffer, width=145, height=145),
        Image(residual_buffer, width=145, height=145),
    ]]

    image_table = Table(
        image_data,
        colWidths=[165, 165, 165]
    )

    image_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story.append(image_table)

    labels = Table(
        [[
            Paragraph("<b>Original</b>", small_style),
            Paragraph("<b>Reconstruction</b>", small_style),
            Paragraph("<b>Residual</b>", small_style),
        ]],
        colWidths=[165, 165, 165]
    )

    labels.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )

    story.append(labels)

    # Reconstruction metrics.
    story.append(
        Paragraph("Reconstruction Metrics", section_style)
    )

    metrics_data = [
        ["Metric", "Value"]
    ]

    for name, value in report["reconstruction_metrics"].items():
        if isinstance(value, (int, float, np.number)):
            metrics_data.append([
                escape(str(name).upper()),
                f"{float(value):.8f}"
            ])

    metrics_table = Table(
        metrics_data,
        colWidths=[250, 250]
    )

    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(metrics_table)

    # Limitations.
    story.append(
        Paragraph("Limitations and Interpretation", section_style)
    )

    for limitation in report["limitations"]:
        story.append(
            Paragraph(
                "&#8226; " + escape(str(limitation)),
                normal_style
            )
        )

    # Optimizer information.
    story.append(
        Paragraph("Optimization Information", section_style)
    )

    optimizer = report["optimizer"]

    optimizer_data = [
        ["Optimizer status", str(optimizer.get("success", "Unknown"))],
        ["Optimizer message", escape(str(optimizer.get("message", "N/A")))],
    ]

    optimizer_table = Table(
        optimizer_data,
        colWidths=[145, 355]
    )

    optimizer_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F6")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(optimizer_table)

    # Footer.
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "Generated by Olin | Gravitational Lens Inference",
            ParagraphStyle(
                "OlinFooter",
                parent=small_style,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
        )
    )

    # Assemble and save the PDF.
    doc.build(story)

    print("PDF report saved:", output_path)

    return output_path

