"""Main Streamlit web application for the AI PDF-to-Word Document Automation Agent."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from doc_generator import inspect_template_variables, populate_word_template
from extractor import (
    PROVIDER_CONFIGS,
    detect_provider_from_key_or_model,
    extract_demo_fallback,
    extract_structured_data,
    extract_text_from_pdf,
    fetch_available_models,
)
from sample_generator import generate_sample_files_bundle
from schemas import DocumentData, FIELD_LABELS

load_dotenv()


def _inject_responsive_css() -> None:
    """Inject responsive CSS so the app looks clean and works identically on Web & Mobile."""
    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: #ffffff;
            padding: 1.4rem 1.6rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        }
        .main-header h1 {
            margin: 0;
            font-size: 1.65rem;
            font-weight: 700;
            color: #ffffff !important;
        }
        .main-header p {
            margin: 0.4rem 0 0 0;
            font-size: 0.96rem;
            color: #cbd5e1;
        }
        .var-badge {
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 0.22rem 0.55rem;
            margin: 0.18rem;
            font-family: monospace;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .privacy-box {
            background: #f0fdf4;
            border-left: 4px solid #16a34a;
            color: #14532d;
            padding: 0.75rem 0.9rem;
            border-radius: 6px;
            font-size: 0.86rem;
            margin-top: 0.75rem;
        }
        @media (max-width: 768px) {
            .main-header {
                padding: 1rem 1.1rem;
            }
            .main-header h1 {
                font-size: 1.25rem;
            }
            .main-header p {
                font-size: 0.85rem;
            }
            .stButton > button, .stDownloadButton > button {
                width: 100% !important;
                min-height: 2.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Initialize required session state variables."""
    defaults: dict[str, Any] = {
        "pdf_bytes": None,
        "pdf_name": None,
        "pdf_meta": None,
        "docx_bytes": None,
        "docx_name": None,
        "template_vars": set(),
        "extracted_data": None,
        "using_sample_files": False,
        "live_models": {},
    }
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def _load_sample_files_into_state() -> None:
    """Generate sample PDF and Word template in memory and store in session_state."""
    pdf_bytes, docx_bytes = generate_sample_files_bundle()
    st.session_state["pdf_bytes"] = pdf_bytes
    st.session_state["pdf_name"] = "sample_title_report.pdf"
    st.session_state["pdf_meta"] = extract_text_from_pdf(pdf_bytes)

    st.session_state["docx_bytes"] = docx_bytes
    st.session_state["docx_name"] = "sample_closing_template.docx"
    st.session_state["template_vars"] = inspect_template_variables(docx_bytes)
    st.session_state["using_sample_files"] = True


def _render_sidebar() -> tuple[str, str, str, bool]:
    """Render sidebar controls and return (provider_key, api_key, selected_model, allow_demo_mode)."""
    with st.sidebar:
        st.header("⚙️ Configuration & AI Provider")

        provider_keys = ["gemini", "groq", "openai"]
        provider_key = st.selectbox(
            "AI Provider (Free & Paid Options)",
            options=provider_keys,
            format_func=lambda k: PROVIDER_CONFIGS[k]["label"],
            index=0,
            help="Choose Google Gemini or Groq for 100% free API keys, or OpenAI.",
        )

        p_cfg = PROVIDER_CONFIGS[provider_key]
        st.caption(f"🔑 [Get a {p_cfg['label']} API Key here]({p_cfg['key_url']})")

        env_api_key = (
            os.getenv(p_cfg["env_var"], "")
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GROQ_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )

        placeholder_map = {
            "gemini": "AIzaSy...",
            "groq": "gsk_...",
            "openai": "sk-...",
        }

        api_key_input = st.text_input(
            f"{p_cfg['label']} API Key",
            value=env_api_key,
            type="password",
            placeholder=placeholder_map.get(provider_key, "Enter API Key..."),
            help="Paste your API key (AIza... for Gemini Free, gsk_... for Groq Free, or sk-... for OpenAI).",
        ).strip()

        eff_provider, _ = detect_provider_from_key_or_model(
            api_key=api_key_input,
            model="",
            explicit_provider=provider_key,
        )

        if api_key_input and st.button(
            "🔄 Kiểm tra API Key & Tải danh sách Model",
            use_container_width=True,
        ):
            with st.spinner("Đang kiểm tra API Key và lấy danh sách model khả dụng..."):
                discovered = fetch_available_models(api_key_input, provider=eff_provider)
                if discovered:
                    st.session_state["live_models"][eff_provider] = discovered
                    st.success(
                        f"Kết nối thành công! Tìm thấy {len(discovered)} model khả dụng cho tài khoản của bạn."
                    )
                else:
                    st.warning(
                        "Không lấy được danh sách model. Vui lòng kiểm tra lại mã API Key của bạn."
                    )

        model_options = (
            st.session_state.get("live_models", {}).get(eff_provider)
            or PROVIDER_CONFIGS[eff_provider]["models"]
        )

        selected_model = st.selectbox(
            "LLM Extraction Model (Auto-Fallbacks on 404)",
            options=model_options,
            index=0,
            help="Even if a model is retired, the system automatically discovers and uses active models on your account.",
        )

        allow_demo_mode = st.checkbox(
            "Enable Offline Demo Fallback (if no API key)",
            value=True,
            help="Allows testing the end-to-end pipeline with sample documents even without an API key.",
        )

        st.divider()
        st.subheader("🧪 Instant Portfolio Demo")
        st.caption(
            "Don't have a PDF and `.docx` template handy? Generate a realistic Real Estate "
            "Title Report PDF and matching Jinja2 Word template in one click."
        )
        if st.button("✨ Generate Sample Test Files", use_container_width=True, type="secondary"):
            try:
                _load_sample_files_into_state()
                st.success("Sample PDF & Word template loaded into memory!")
            except Exception as exc:
                st.error(f"Failed to generate sample files: {exc}")

        if st.session_state.get("using_sample_files") and st.session_state.get("pdf_bytes"):
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Sample PDF",
                    data=st.session_state["pdf_bytes"],
                    file_name="sample_title_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            with col_dl2:
                st.download_button(
                    "📥 Sample .docx",
                    data=st.session_state["docx_bytes"],
                    file_name="sample_closing_template.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                )

        st.divider()
        st.subheader("🏗️ System Architecture")
        st.markdown(
            "1. **PDF Ingestion:** `PyMuPDF` reads pages in memory (+ optional OCR fallback).\n"
            "2. **Structured LLM Parse:** `Gemini / Groq / OpenAI` enforces strict `Pydantic` schema (`DocumentData`).\n"
            "3. **HITL Review:** Interactive verification table & mobile form before document creation.\n"
            "4. **Word Rendering:** `docxtpl` populates `{{ variables }}` with 0% formatting loss."
        )

        st.markdown(
            '<div class="privacy-box">'
            "🔒 <strong>Zero Disk Persistence Notice:</strong><br/>"
            "Files are processed strictly in memory (<code>BytesIO</code>) and never stored on disk."
            "</div>",
            unsafe_allow_html=True,
        )

    return eff_provider, api_key_input, selected_model, allow_demo_mode


def main() -> None:
    """Run the Streamlit 3-step workflow UI."""
    st.set_page_config(
        page_title="AI PDF-to-Word Document Automation Agent",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_responsive_css()
    _init_session_state()

    provider_key, api_key, selected_model, allow_demo_mode = _render_sidebar()

    st.markdown(
        """
        <div class="main-header">
            <h1>📄 AI PDF-to-Word Document Automation Agent</h1>
            <p>Extract structured entities from unstructured PDFs with LLM Structured Outputs (Free Gemini, Free Groq, or OpenAI), review via Human-in-the-Loop, and populate formatted Word (.docx) templates automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # STEP 1: DOCUMENT UPLOAD (2 COLUMNS)
    # =========================================================================
    st.subheader("Step 1: Upload Source PDF & Target Word Template")
    col_pdf, col_docx = st.columns(2, gap="large")

    with col_pdf:
        st.markdown("##### 1A. Source Unstructured PDF (`.pdf`)")
        uploaded_pdf = st.file_uploader(
            "Upload Source PDF Document",
            type=["pdf"],
            key="pdf_uploader",
            help="Upload a contract, real estate title report, or invoice in PDF format.",
        )

        if uploaded_pdf is not None:
            try:
                raw_pdf_bytes = uploaded_pdf.getvalue()
                pdf_meta = extract_text_from_pdf(raw_pdf_bytes)
                st.session_state["pdf_bytes"] = raw_pdf_bytes
                st.session_state["pdf_name"] = uploaded_pdf.name
                st.session_state["pdf_meta"] = pdf_meta
                st.session_state["using_sample_files"] = False
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["pdf_meta"] is not None:
            meta = st.session_state["pdf_meta"]
            st.success(
                f"Loaded PDF: **{st.session_state['pdf_name']}** "
                f"({meta['page_count']} {'page' if meta['page_count'] == 1 else 'pages'})"
            )
            if meta.get("is_scanned_warning"):
                st.warning(
                    "⚠️ Extracted text is under 50 characters. This PDF appears to be an image scan "
                    "without embedded text. Ensure OCR (`pytesseract`) is installed or upload a text-searchable PDF."
                )
            with st.expander("🔍 Preview Extracted Raw PDF Text Snippet", expanded=False):
                preview_text = meta["full_text"]
                st.text_area(
                    "Extracted Text Preview",
                    value=preview_text[:3000] + ("...\n[Truncated]" if len(preview_text) > 3000 else ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                )
        else:
            st.info("Upload a `.pdf` file above or click **Generate Sample Test Files** in the sidebar.")

    with col_docx:
        st.markdown("##### 1B. Target Word Template (`.docx`)")
        uploaded_docx = st.file_uploader(
            "Upload Word (.docx) Template with {{ tags }}",
            type=["docx"],
            key="docx_uploader",
            help="Upload a Microsoft Word (.docx) file containing Jinja2 tags like {{ client_name }}.",
        )

        if uploaded_docx is not None:
            try:
                raw_docx_bytes = uploaded_docx.getvalue()
                detected_vars = inspect_template_variables(raw_docx_bytes)
                st.session_state["docx_bytes"] = raw_docx_bytes
                st.session_state["docx_name"] = uploaded_docx.name
                st.session_state["template_vars"] = detected_vars
                st.session_state["using_sample_files"] = False
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["docx_bytes"] is not None:
            detected_vars: set[str] = st.session_state["template_vars"]
            st.success(
                f"Loaded Template: **{st.session_state['docx_name']}** "
                f"({len(detected_vars)} Jinja2 {'variable' if len(detected_vars) == 1 else 'variables'} detected)"
            )
            if detected_vars:
                badges_html = "".join(
                    f'<span class="var-badge">{{{{ {var} }}}}</span>'
                    for var in sorted(detected_vars)
                )
                st.markdown("**Detected Template Placeholders:**", unsafe_allow_html=True)
                st.markdown(badges_html, unsafe_allow_html=True)
            else:
                st.warning(
                    "⚠️ No `{{ variable_name }}` placeholders were found in this Word template. "
                    "Add tags such as `{{ client_name }}` inside your `.docx` file."
                )
        else:
            st.info("Upload a `.docx` template above or click **Generate Sample Test Files** in the sidebar.")

    st.divider()

    # =========================================================================
    # STEP 2: AI EXTRACTION & HUMAN-IN-THE-LOOP (HITL) VERIFICATION
    # =========================================================================
    st.subheader("Step 2: AI Extraction & Human-in-the-Loop (HITL) Verification")

    pdf_ready = st.session_state["pdf_meta"] is not None
    run_clicked = st.button(
        "🚀 Run AI Extraction Pipeline",
        type="primary",
        disabled=not pdf_ready,
        use_container_width=True,
    )

    if run_clicked:
        pdf_meta = st.session_state["pdf_meta"]
        raw_text = pdf_meta["full_text"] if pdf_meta else ""

        with st.spinner("Running AI extraction pipeline (auto-discovering active models)..."):
            try:
                if api_key:
                    eff_provider, eff_model = detect_provider_from_key_or_model(
                        api_key=api_key,
                        model=selected_model,
                        explicit_provider=provider_key,
                    )
                    extracted_obj = extract_structured_data(
                        text=raw_text,
                        api_key=api_key,
                        model=eff_model,
                        provider=eff_provider,
                    )
                    st.session_state["extracted_data"] = extracted_obj.model_dump()
                    st.success(
                        f"Successfully extracted {len(st.session_state['extracted_data'])} structured fields "
                        f"via **{PROVIDER_CONFIGS[eff_provider]['label']}**!"
                    )
                elif allow_demo_mode:
                    extracted_obj = extract_demo_fallback(raw_text)
                    st.session_state["extracted_data"] = extracted_obj.model_dump()
                    st.info(
                        "ℹ️ Ran in **Offline Demo Extraction Mode** (no API key entered). "
                        "Paste a free Google Gemini or Groq API key in the sidebar anytime for live AI extraction."
                    )
                else:
                    st.error(
                        "API Key is required. Please enter a Free Google Gemini, Free Groq, or OpenAI API key "
                        "in the sidebar, or enable 'Offline Demo Fallback'."
                    )
            except Exception as exc:
                st.error(str(exc))

    if st.session_state["extracted_data"] is not None:
        current_data: dict[str, str] = st.session_state["extracted_data"]

        st.markdown(
            "Review and edit any extracted value below before populating your Word document. "
            "Switch between **Table View** and **Mobile-Friendly Form View** anytime — both stay in sync."
        )

        edit_mode = st.radio(
            "Verification Editor Layout",
            options=["📊 Interactive Table View (st.data_editor)", "📱 Form Input View (Mobile & Detailed Edit)"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if edit_mode.startswith("📊"):
            rows = [
                {
                    "Template Variable": f"{{{{ {field_key} }}}}",
                    "Field Description": FIELD_LABELS.get(field_key, field_key),
                    "Extracted Value (Editable)": str(current_data.get(field_key, "N/A")),
                    "_key": field_key,
                }
                for field_key in DocumentData.model_fields.keys()
            ]
            df = pd.DataFrame(rows)

            edited_df = st.data_editor(
                df[["Template Variable", "Field Description", "Extracted Value (Editable)"]],
                use_container_width=True,
                hide_index=True,
                disabled=["Template Variable", "Field Description"],
                column_config={
                    "Template Variable": st.column_config.TextColumn("Jinja2 Tag", width="medium"),
                    "Field Description": st.column_config.TextColumn("Entity Name", width="medium"),
                    "Extracted Value (Editable)": st.column_config.TextColumn(
                        "Verified Value (Click to Edit)",
                        width="large",
                        required=True,
                    ),
                },
                key="hitl_data_editor",
            )

            updated_dict: dict[str, str] = {}
            for idx, field_key in enumerate(DocumentData.model_fields.keys()):
                val = str(edited_df.iloc[idx]["Extracted Value (Editable)"]).strip()
                updated_dict[field_key] = val if val else "N/A"
            st.session_state["extracted_data"] = updated_dict
        else:
            updated_dict = dict(current_data)
            f_col1, f_col2 = st.columns(2)
            single_line_keys = [
                "client_name",
                "entity_address",
                "reference_id",
                "effective_date",
                "institution_name",
                "total_amount",
            ]
            for idx, field_key in enumerate(single_line_keys):
                target_col = f_col1 if idx % 2 == 0 else f_col2
                with target_col:
                    new_val = st.text_input(
                        f"{FIELD_LABELS[field_key]} (`{{{{ {field_key} }}}}`)",
                        value=current_data.get(field_key, "N/A"),
                        key=f"form_input_{field_key}",
                    )
                    updated_dict[field_key] = new_val.strip() or "N/A"

            for multi_key in ["legal_description", "key_summary"]:
                new_val = st.text_area(
                    f"{FIELD_LABELS[multi_key]} (`{{{{ {multi_key} }}}}`)",
                    value=current_data.get(multi_key, "N/A"),
                    height=90,
                    key=f"form_input_{multi_key}",
                )
                updated_dict[multi_key] = new_val.strip() or "N/A"

            st.session_state["extracted_data"] = updated_dict

    st.divider()

    # =========================================================================
    # STEP 3: EXPORT FINAL WORD DOCUMENT
    # =========================================================================
    st.subheader("Step 3: Export Final Populated Word Document")

    can_export = (
        st.session_state["docx_bytes"] is not None
        and st.session_state["extracted_data"] is not None
    )

    if can_export:
        try:
            populated_docx_bytes = populate_word_template(
                template_bytes=st.session_state["docx_bytes"],
                data_dict=st.session_state["extracted_data"],
            )
            base_name = (st.session_state["docx_name"] or "document.docx").rsplit(".", 1)[0]
            output_filename = f"populated_{base_name}.docx"

            st.success(
                "✅ Document rendered in memory with 100% original Word formatting preserved!"
            )
            st.download_button(
                label="📥 Download Populated Word Document (.docx)",
                data=populated_docx_bytes,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Template rendering error: {exc}")
    else:
        st.info(
            "Complete **Step 1** (load PDF & Word template) and **Step 2** (run AI extraction) "
            "to enable final Word document export."
        )


if __name__ == "__main__":
    main()
