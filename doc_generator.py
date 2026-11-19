"""Word template inspection and Jinja2 rendering engine using docxtpl."""

from __future__ import annotations

import io
from typing import Any

from docxtpl import DocxTemplate


def inspect_template_variables(template_bytes: bytes) -> set[str]:
    """
    Detect all undeclared Jinja2 {{ variable_name }} placeholders in a .docx template.

    Args:
        template_bytes: Raw bytes of the uploaded .docx template file.

    Returns:
        A set of detected variable names.
    """
    if not template_bytes:
        raise ValueError("The uploaded Word template file is empty.")

    try:
        doc = DocxTemplate(io.BytesIO(template_bytes))
        variables = doc.get_undeclared_template_variables()
        return set(variables) if variables else set()
    except Exception as exc:
        raise ValueError(
            f"Unable to parse Word template (.docx). Check that the file is a valid Word document "
            f"and all '{{{{ variable }}}}' tags are properly formatted. ({exc})"
        ) from exc


def populate_word_template(template_bytes: bytes, data_dict: dict[str, Any]) -> bytes:
    """
    Populate a .docx template with extracted dictionary values in memory while preserving 100% formatting.

    Args:
        template_bytes: Raw bytes of the source .docx template.
        data_dict: Dictionary mapping template variable names to string values.

    Returns:
        Rendered .docx file as raw bytes.
    """
    if not template_bytes:
        raise ValueError("Word template bytes cannot be empty.")

    try:
        doc = DocxTemplate(io.BytesIO(template_bytes))
        doc.render(data_dict, autoescape=True)
        output_buffer = io.BytesIO()
        doc.save(output_buffer)
        output_buffer.seek(0)
        return output_buffer.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Failed to render Word document from template: {exc}") from exc
