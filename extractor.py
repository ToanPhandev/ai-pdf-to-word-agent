"""PDF text parsing and Multi-Provider (Free Google Gemini, Free Groq, OpenAI) Structured Outputs module."""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]
import requests
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

from schemas import DocumentData

SYSTEM_PROMPT = (
    "You are an expert legal, financial, and real estate document extraction agent. "
    "Extract the requested fields accurately from the provided raw PDF text. "
    "Follow these strict rules:\n"
    "1. Standardize any dates to YYYY-MM-DD format.\n"
    "2. Format monetary amounts with currency symbols and commas (e.g., $485,000.00).\n"
    "3. Keep 'key_summary' to a concise 2-sentence executive summary.\n"
    "4. If any field cannot be reliably found in the source document, return 'N/A'. "
    "Never invent or hallucinate missing information.\n"
    "5. Respond ONLY with a valid JSON object containing these exact 8 keys: "
    "client_name, entity_address, reference_id, effective_date, institution_name, "
    "total_amount, legal_description, key_summary."
)

PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "gemini": {
        "label": "Google Gemini (100% Free)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        "env_var": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "groq": {
        "label": "Groq Cloud (100% Free)",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
        ],
        "env_var": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
    },
    "openai": {
        "label": "OpenAI (Paid)",
        "base_url": None,
        "models": ["gpt-4o-mini", "gpt-4o"],
        "env_var": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
    },
}


def detect_provider_from_key_or_model(
    api_key: str,
    model: str,
    explicit_provider: str | None = None,
) -> tuple[str, str]:
    """
    Determine the provider ('gemini', 'groq', or 'openai') from the API key prefix or model.
    """
    cleaned_key = (api_key or "").strip()

    if cleaned_key.startswith("AIza"):
        provider = "gemini"
    elif cleaned_key.startswith("gsk_"):
        provider = "groq"
    elif cleaned_key.startswith("sk-"):
        provider = "openai"
    elif explicit_provider in PROVIDER_CONFIGS:
        provider = explicit_provider
    elif model.startswith("gemini"):
        provider = "gemini"
    elif any(k in model for k in ("llama", "mixtral", "qwen", "gpt-oss")):
        provider = "groq"
    else:
        provider = "openai"

    valid_models = PROVIDER_CONFIGS[provider]["models"]
    resolved_model = model if model in valid_models else valid_models[0]
    return provider, resolved_model


def fetch_available_models(api_key: str, provider: str | None = None) -> list[str]:
    """
    Dynamically query the provider's /models endpoint to discover exact models enabled for this API key.
    """
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        return []

    resolved_provider, _ = detect_provider_from_key_or_model(
        api_key=cleaned_key,
        model="",
        explicit_provider=provider,
    )

    if resolved_provider == "gemini":
        discovered: list[str] = []
        for api_ver in ("v1beta", "v1"):
            try:
                resp = requests.get(
                    f"https://generativelanguage.googleapis.com/{api_ver}/models",
                    params={"key": cleaned_key},
                    timeout=12,
                )
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                for item in payload.get("models", []):
                    methods = item.get("supportedGenerationMethods", [])
                    raw_name = item.get("name", "")
                    short_name = raw_name.replace("models/", "")
                    lower_name = short_name.lower()
                    if "generateContent" not in methods:
                        continue
                    if "gemini" not in lower_name:
                        continue
                    if any(
                        skip in lower_name
                        for skip in ("embedding", "imagen", "veo", "aqa", "tts", "audio", "vision", "computer-use")
                    ):
                        continue
                    if short_name not in discovered:
                        discovered.append(short_name)
                if discovered:
                    break
            except Exception:
                continue

        # Sort preferred text/flash models first
        def _gemini_rank(m: str) -> tuple[int, str]:
            ml = m.lower()
            if "2.5-flash" in ml and "lite" not in ml and "preview" not in ml:
                return (0, m)
            if "2.0-flash" in ml and "lite" not in ml:
                return (1, m)
            if "flash" in ml and "exp" not in ml:
                return (2, m)
            if "pro" in ml and "exp" not in ml:
                return (3, m)
            return (4, m)

        discovered.sort(key=_gemini_rank)
        return discovered

    if resolved_provider == "groq":
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {cleaned_key}"},
                timeout=12,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
            groq_models: list[str] = []
            for item in payload.get("data", []):
                if item.get("active", True) is False:
                    continue
                m_id = str(item.get("id", "")).strip()
                ml = m_id.lower()
                if not m_id:
                    continue
                if any(
                    skip in ml
                    for skip in ("whisper", "tts", "playai", "guard", "orpheus", "embed", "safeguard", "allam")
                ):
                    continue
                groq_models.append(m_id)

            def _groq_rank(m: str) -> tuple[int, str]:
                ml = m.lower()
                if "llama-3.3" in ml or "llama-4" in ml:
                    return (0, m)
                if "gpt-oss" in ml:
                    return (1, m)
                if "llama-3.1" in ml:
                    return (2, m)
                if "qwen" in ml or "gemma" in ml:
                    return (3, m)
                return (4, m)

            groq_models.sort(key=_groq_rank)
            return groq_models
        except Exception:
            return []

    return PROVIDER_CONFIGS["openai"]["models"]


def _parse_json_to_document_data(raw_text: str) -> DocumentData:
    """Safely extract and parse a JSON object from LLM text response into DocumentData."""
    cleaned = (raw_text or "").strip()
    # Strip markdown code fences if present
    if "```" in cleaned:
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("{"):
        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0).strip()

    raw_dict = json.loads(cleaned)
    if not isinstance(raw_dict, dict):
        raise ValueError("Model response did not parse into a JSON object.")

    normalized = {
        field_name: str(raw_dict.get(field_name) or "N/A").strip() or "N/A"
        for field_name in DocumentData.model_fields.keys()
    }
    return DocumentData(**normalized)


def _extract_with_gemini_native(text: str, api_key: str, preferred_model: str) -> DocumentData:
    """
    Call Google Gemini Native REST API (generateContent), automatically discovering
    active models on the user's account to prevent 404 Not Found errors.
    """
    live_models = fetch_available_models(api_key, provider="gemini")
    candidates: list[str] = []

    if preferred_model and (not live_models or preferred_model in live_models):
        candidates.append(preferred_model)
    for m in live_models:
        if m not in candidates:
            candidates.append(m)
    for fallback_m in PROVIDER_CONFIGS["gemini"]["models"]:
        if fallback_m not in candidates:
            candidates.append(fallback_m)

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        "Extract the structured document fields from the following PDF text and return ONLY "
        "a JSON object with keys: client_name, entity_address, reference_id, effective_date, "
        "institution_name, total_amount, legal_description, key_summary.\n\n"
        f"{text}"
    )

    last_error_msg = "No working Gemini model found for this API key."

    for candidate_model in candidates:
        clean_model_id = candidate_model.replace("models/", "")
        for api_ver in ("v1beta", "v1"):
            url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{clean_model_id}:generateContent"
            body: dict[str, Any] = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.0},
            }
            if api_ver == "v1beta":
                body["generationConfig"]["responseMimeType"] = "application/json"

            try:
                resp = requests.post(
                    url,
                    params={"key": api_key},
                    json=body,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_error_msg = f"Network error connecting to Google Gemini API: {exc}"
                continue

            if resp.status_code == 200:
                data = resp.json()
                try:
                    parts = data["candidates"][0]["content"]["parts"]
                    text_out = "".join(p.get("text", "") for p in parts if "text" in p)
                    return _parse_json_to_document_data(text_out)
                except Exception as exc:
                    last_error_msg = f"Failed to parse Gemini JSON output from {clean_model_id}: {exc}"
                    continue

            # Parse error response
            try:
                err_detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_detail = resp.text

            if resp.status_code in (401, 403) or "API_KEY_INVALID" in err_detail:
                raise RuntimeError(
                    f"Google Gemini Authentication Error: Your API key is invalid or unauthorized. ({err_detail})"
                )

            if resp.status_code == 404:
                last_error_msg = f"Model '{clean_model_id}' not found on {api_ver} ({err_detail})"
                continue

            if resp.status_code == 429:
                last_error_msg = f"Quota/Rate limit reached on '{clean_model_id}' ({err_detail})"
                continue

            last_error_msg = f"Google Gemini API ({clean_model_id}) returned HTTP {resp.status_code}: {err_detail}"

    raise RuntimeError(
        f"Google Gemini API extraction failed across available models. Last error: {last_error_msg}"
    )


def _extract_with_groq_native(text: str, api_key: str, preferred_model: str) -> DocumentData:
    """
    Call Groq Cloud REST API, automatically discovering active models enabled for the user's
    Groq API key and falling back seamlessly if a selected model returns 404.
    """
    live_models = fetch_available_models(api_key, provider="groq")
    candidates: list[str] = []

    if preferred_model and (not live_models or preferred_model in live_models):
        candidates.append(preferred_model)
    for m in live_models:
        if m not in candidates:
            candidates.append(m)
    for fallback_m in PROVIDER_CONFIGS["groq"]["models"]:
        if fallback_m not in candidates:
            candidates.append(fallback_m)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract the structured document fields from the following PDF text and return ONLY "
                "a valid JSON object with keys: client_name, entity_address, reference_id, effective_date, "
                "institution_name, total_amount, legal_description, key_summary.\n\n"
                f"{text}"
            ),
        },
    ]

    last_error_msg = "No working Groq model found for this API key."

    for candidate_model in candidates:
        for use_json_mode in (True, False):
            payload: dict[str, Any] = {
                "model": candidate_model,
                "messages": messages,
                "temperature": 0.0,
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}

            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_error_msg = f"Network error connecting to Groq API: {exc}"
                break

            if resp.status_code == 200:
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return _parse_json_to_document_data(content)
                except Exception as exc:
                    last_error_msg = f"Failed to parse JSON from Groq model {candidate_model}: {exc}"
                    continue

            try:
                err_detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err_detail = resp.text

            if resp.status_code == 401:
                raise RuntimeError(
                    f"Groq Authentication Error: Your Groq API Key is invalid or expired. ({err_detail})"
                )

            if resp.status_code == 404:
                last_error_msg = f"Model '{candidate_model}' not available ({err_detail})"
                break  # Move to next candidate model

            if resp.status_code == 400 and use_json_mode:
                # Some models reject response_format={"type": "json_object"}, retry once without it
                continue

            last_error_msg = f"Groq API ({candidate_model}) returned HTTP {resp.status_code}: {err_detail}"
            break

    raise RuntimeError(
        f"Groq Cloud API extraction failed across available models. Last error: {last_error_msg}"
    )


def _attempt_ocr_fallback(doc: fitz.Document) -> str:
    """Attempt optional OCR fallback with pytesseract and PIL if installed."""
    try:
        import io
        import pytesseract
        from PIL import Image

        ocr_pages: list[str] = []
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img).strip()
            ocr_pages.append(f"--- PAGE {idx} (OCR) ---\n{page_text}")
        return "\n\n".join(ocr_pages).strip()
    except Exception:
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """
    Extract text page-by-page from raw PDF bytes in memory.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        Dictionary with keys:
            - full_text (str): Combined text across all pages with page headers.
            - page_count (int): Total number of pages in the PDF.
            - is_scanned_warning (bool): True if extracted text length is < 50 chars.
            - used_ocr (bool): True if optional OCR fallback recovered text.
    """
    if not pdf_bytes:
        raise ValueError("The uploaded PDF file is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to read PDF file. Ensure it is a valid PDF document. ({exc})") from exc

    try:
        page_count = len(doc)
        if page_count == 0:
            raise ValueError("The PDF document contains 0 pages.")

        pages_text: list[str] = []
        raw_content_chars = 0

        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            raw_content_chars += len(text)
            pages_text.append(f"--- PAGE {page_num + 1} ---\n{text}")

        full_text = "\n\n".join(pages_text).strip()
        used_ocr = False

        if raw_content_chars < 50:
            ocr_text = _attempt_ocr_fallback(doc)
            if len(ocr_text) >= 50:
                full_text = ocr_text
                raw_content_chars = len(ocr_text)
                used_ocr = True

        is_scanned_warning = raw_content_chars < 50

        return {
            "full_text": full_text,
            "page_count": page_count,
            "is_scanned_warning": is_scanned_warning,
            "used_ocr": used_ocr,
        }
    finally:
        doc.close()


def extract_structured_data(
    text: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    provider: str | None = None,
) -> DocumentData:
    """
    Extract structured DocumentData from raw PDF text using Free Google Gemini, Free Groq, or OpenAI.
    Automatically discovers active models for Gemini and Groq keys to avoid 404 Not Found errors.
    """
    cleaned_key = (api_key or "").strip()
    if not cleaned_key:
        raise ValueError(
            "API Key is missing. Please enter a Free Google Gemini API key, Free Groq API key, "
            "or OpenAI API key in the sidebar."
        )

    if not text or not text.strip():
        raise ValueError("No text content available to extract from the PDF.")

    resolved_provider, resolved_model = detect_provider_from_key_or_model(
        api_key=cleaned_key,
        model=model,
        explicit_provider=provider,
    )

    if resolved_provider == "gemini":
        return _extract_with_gemini_native(text=text, api_key=cleaned_key, preferred_model=resolved_model)

    if resolved_provider == "groq":
        return _extract_with_groq_native(text=text, api_key=cleaned_key, preferred_model=resolved_model)

    # OpenAI Provider path
    client = OpenAI(api_key=cleaned_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract the structured document fields from the following PDF text:\n\n"
                f"{text}"
            ),
        },
    ]

    try:
        completion = client.beta.chat.completions.parse(
            model=resolved_model,
            temperature=0.0,
            messages=messages,
            response_format=DocumentData,
        )
        parsed: DocumentData | None = completion.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("The OpenAI model did not return a valid structured response.")
        return parsed
    except AuthenticationError as exc:
        raise RuntimeError(
            "OpenAI Authentication failed: Your OpenAI API Key is invalid or expired. "
            "Please verify your key in the sidebar."
        ) from exc
    except RateLimitError as exc:
        raise RuntimeError(
            "OpenAI rate limit or quota exceeded. Please check your OpenAI billing/quota."
        ) from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI API error occurred: {exc.message}") from exc
    except Exception as exc:
        if isinstance(exc, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Unexpected error during OpenAI extraction: {exc}") from exc


def extract_demo_fallback(text: str) -> DocumentData:
    """
    Deterministic offline parser for quick portfolio demonstration when no API key is provided.
    Attempts regex pattern extraction from the PDF text and falls back to 'N/A'.
    """
    def _find(pattern: str, default: str = "N/A") -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default

    client_name = _find(r"(?:Primary\s+Owner\s*/\s*Buyer|Client\s+Name|Owner|Buyer|Borrower)\s*:\s*(.+)")
    entity_address = _find(r"(?:Property\s+Address|Subject\s+Property|Entity\s+Address|Address)\s*:\s*(.+)")
    reference_id = _find(r"(?:Assessor\s+Parcel\s+No\.?\s*\(APN\)|APN|Reference\s+ID|Policy\s+No\.?)\s*:\s*(.+)")
    effective_date = _find(r"(?:Effective\s*/\s*Recording\s+Date|Effective\s+Date|Recording\s+Date|Issue\s+Date)\s*:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})")
    institution_name = _find(r"(?:Lender\s*/\s*Title\s+Institution|Title\s+Company|Lender|Institution)\s*:\s*(.+)")
    total_amount = _find(r"(?:Total\s+Loan\s*/\s+Valuation\s+Amount|Loan\s+Amount|Total\s+Amount|Valuation)\s*:\s*(\$[0-9,]+(?:\.[0-9]{2})?)")

    legal_match = re.search(
        r"Legal\s+Description\s*:\s*(.+?)(?=\n[0-9]+\.\s+[A-Z]|\nLender|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    legal_description = (
        " ".join(legal_match.group(1).split()) if legal_match else _find(r"(?:Legal\s+Description)\s*:\s*(.+)")
    )

    summary_match = re.search(
        r"(?:Executive\s+Summary|Summary\s+Notes)\s*:\s*(.+?)(?=\n[0-9]+\.\s+[A-Z]|\nGenerated by|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    key_summary = (
        " ".join(summary_match.group(1).split()) if summary_match else _find(r"(?:Executive\s+Summary|Summary\s+Notes)\s*:\s*(.+)")
    )

    if key_summary == "N/A" and client_name != "N/A":
        key_summary = (
            f"Title report confirms {client_name} as the vested party for the property at {entity_address}. "
            f"Transaction is underwritten by {institution_name} with a total valuation of {total_amount}."
        )

    return DocumentData(
        client_name=client_name,
        entity_address=entity_address,
        reference_id=reference_id,
        effective_date=effective_date,
        institution_name=institution_name,
        total_amount=total_amount,
        legal_description=legal_description,
        key_summary=key_summary,
    )
