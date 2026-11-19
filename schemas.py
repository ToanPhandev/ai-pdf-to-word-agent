"""Pydantic data models for structured LLM extraction from PDF documents."""

from pydantic import BaseModel, Field


class DocumentData(BaseModel):
    """Structured schema representing key entities extracted from a document."""

    client_name: str = Field(
        default="N/A",
        description=(
            "Full name of the primary client, buyer, borrower, or property owner. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    entity_address: str = Field(
        default="N/A",
        description=(
            "Full street address, city, state, and ZIP code of the subject property or entity. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    reference_id: str = Field(
        default="N/A",
        description=(
            "Parcel ID, Assessor's Parcel Number (APN), Policy Number, or Contract Reference Number. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    effective_date: str = Field(
        default="N/A",
        description=(
            "Effective, issue, or recording date standardized to YYYY-MM-DD format. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    institution_name: str = Field(
        default="N/A",
        description=(
            "Lender, bank, title company, or issuing organization name. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    total_amount: str = Field(
        default="N/A",
        description=(
            "Monetary amount, loan value, purchase price, or valuation formatted with currency symbol (e.g., $485,000.00). "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    legal_description: str = Field(
        default="N/A",
        description=(
            "Brief property legal description or contract scope description if available. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )
    key_summary: str = Field(
        default="N/A",
        description=(
            "A concise 2-sentence summary of the key terms, findings, or obligations in the document. "
            "Return 'N/A' if not reliably found in the source document."
        ),
    )


FIELD_LABELS: dict[str, str] = {
    "client_name": "Client / Owner Name",
    "entity_address": "Property / Entity Address",
    "reference_id": "Reference / Parcel ID (APN)",
    "effective_date": "Effective / Recording Date (YYYY-MM-DD)",
    "institution_name": "Institution / Lender / Title Co.",
    "total_amount": "Total Monetary Amount",
    "legal_description": "Legal / Scope Description",
    "key_summary": "Key Executive Summary (2 Sentences)",
}
