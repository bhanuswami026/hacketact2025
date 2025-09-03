# app_agentic.py
"""Wesco Journal Totals Validator — Agentic Refactor

This refactor introduces a minimal agentic pipeline (no external agent libs) while
preserving UI text, prompts, parsing, and numerical behavior exactly as in the
original script.

Steps executed by the agent (in order):
1) prepare_image_state
2) vision_extract_totals
3) extract_journal_table
4) compute_excel_totals

Why agentic: Each step is an encapsulated capability that reads/writes a shared
state. The Agent executes steps sequentially, capturing errors and returning the
final state for the Streamlit layer to render. This keeps functionality intact
and auditable while enabling future orchestration (branching/retries) without
changing UI behavior.
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import load_workbook
from PIL import Image

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = "https://openai-mavericks.openai.azure.com/openai/v1/"
deployment_name = "gpt-4"

client = OpenAI(base_url=endpoint, api_key=api_key)

st.set_page_config(page_title="Wesco Journal Totals Validator", layout="wide")


# ---------- Agentic Framework ----------
@dataclass
class Step:
    """A single agent step. Encapsulates a capability.

    Only essential doc to explain the *why*: Enables structured, auditable
    orchestration without altering business logic.
    """

    name: str
    run: Callable[[Dict[str, Any]], Dict[str, Any]]


class Agent:
    """Sequential executor over named steps with shared state dict."""

    def __init__(self, steps: List[Step]):
        self.steps = steps

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state.setdefault("errors", [])
        for step in self.steps:
            try:
                updates = step.run(state)
                if updates:
                    state.update(updates)
            except Exception as exc:
                # *Why*: Capture step-scoped failures but allow UI to decide how to surface them
                state["errors"].append(f"{step.name}: {exc}")
        return state


# ---------- Original Logic — now modularized as steps ----------

def prepare_image_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Read uploaded image file into bytes and base64 for vision request."""
    image_file = state.get("image_file")
    if image_file is None:
        return {}
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {"image_bytes": image_bytes, "image_b64": image_b64}


def vision_extract_totals(state: Dict[str, Any]) -> Dict[str, Any]:
    """Call Azure OpenAI (same prompt and parsing) to extract TOTAL Debit/Credit from image."""
    image_b64 = state.get("image_b64")
    if not image_b64:
        return {"image_debit": None, "image_credit": None}

    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "From this journal table image, extract ONLY the final TOTAL Debit and final TOTAL Credit "
                    "values from the summary row at the bottom of the table. Do not extract line items. "
                    "Return just the two numbers."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        max_tokens=200,
    )

    vision_response = completion.choices[0].message.content or ""
    m = re.findall(r"\$?([\d,]+(?:\.\d{2}))", vision_response)
    if len(m) >= 2:
        image_debit = float(m[0].replace(",", ""))
        image_credit = float(m[1].replace(",", ""))
        return {"image_debit": image_debit, "image_credit": image_credit}
    else:
        # *Why*: Keep exact failure semantics for UI
        return {"image_debit": None, "image_credit": None, "vision_error": True}


def load_full_excel_as_df(filepath) -> pd.DataFrame:
    """Load entire Excel (xls/xlsx) into a DataFrame — mirrors original behavior."""
    ext = Path(filepath.name).suffix.lower()
    if ext == ".xlsx":
        wb = load_workbook(filepath, data_only=True)
        sheet = wb.active
        data = [[cell for cell in row] for row in sheet.iter_rows(values_only=True)]
        return pd.DataFrame(data)
    elif ext == ".xls":
        df_temp = pd.read_excel(filepath, engine="xlrd")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_temp.to_excel(writer, index=False)
        buffer.seek(0)
        wb = load_workbook(buffer, data_only=True)
        sheet = wb.active
        data = [[cell for cell in row] for row in sheet.iter_rows(values_only=True)]
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Only .xls and .xlsx are supported.")


def find_journal_header_row(df_raw: pd.DataFrame) -> Optional[int]:
    """Locate header row that contains both 'Entered Debit' and 'Entered Credit'."""
    for idx, row in df_raw.iterrows():
        row_strs = [str(c).strip().lower() for c in row if pd.notna(c)]
        if any("entered debit" in c for c in row_strs) and any("entered credit" in c for c in row_strs):
            return idx
    return None


def extract_journal_table_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract journal table DataFrame from uploaded Excel, preserving original rules."""
    excel_file = state.get("excel_file")
    if excel_file is None:
        return {"df": pd.DataFrame()}

    df_raw = load_full_excel_as_df(excel_file)
    header_row = find_journal_header_row(df_raw)
    if header_row is None:
        return {"df": pd.DataFrame()}

    df_journal = df_raw.iloc[header_row:].copy()
    df_journal.columns = df_journal.iloc[0]
    df_journal = df_journal.drop(header_row)
    df_journal = df_journal.reset_index(drop=True)
    df_journal.columns = [str(col).strip() for col in df_journal.columns]
    return {"df": df_journal}


def compute_excel_totals_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Excel totals using same filtering, coercion, and summing logic."""
    df = state.get("df")
    if df is None or df.empty:
        return {"excel_debit": None, "excel_credit": None, "debit_col": None, "credit_col": None}

    debit_col = [col for col in df.columns if "entered debit" in str(col).lower()]
    credit_col = [col for col in df.columns if "entered credit" in str(col).lower()]

    if not debit_col or not credit_col:
        return {"excel_debit": None, "excel_credit": None, "debit_col": None, "credit_col": None}

    debit_col = debit_col[0]
    credit_col = credit_col[0]

    for col in [debit_col, credit_col]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce"
        )

    not_both = ~(df[debit_col].notna() & df[credit_col].notna())
    row_text = df.astype(str).apply(lambda r: " | ".join(r.values).lower(), axis=1)
    # Intentionally preserve the original regex (with double backslashes)
    not_summary = ~row_text.str.contains(r"\\btotal\\b|\\bsummary\\b", regex=True)

    context_cols = [
        col for col in df.columns if any(k in str(col).lower() for k in [
            "legal entity", "operating group", "account", "department", "site",
            "intercompany", "project", "future 1", "future 2", "currency"
        ])
    ]

    required_non_nulls = max(1, len(context_cols) // 2)
    not_mostly_empty = df[context_cols].notna().sum(axis=1) >= required_non_nulls if context_cols else True

    filtered = df[not_both & not_summary & not_mostly_empty]
    excel_debit = filtered[debit_col].sum(skipna=True)
    excel_credit = filtered[credit_col].sum(skipna=True)

    return {
        "excel_debit": float(excel_debit) if pd.notna(excel_debit) else None,
        "excel_credit": float(excel_credit) if pd.notna(excel_credit) else None,
        "debit_col": debit_col,
        "credit_col": credit_col,
    }


# ---------- UI (unchanged content/wording) ----------
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown(
        """
        <h1 style='margin-bottom: 0;'>📊 Wesco Journal Totals Validator</h1>
        <p style='margin-top: 0;'>Upload a journal table image and Excel file. We'll extract the <b>TOTAL Debit</b> and <b>TOTAL Credit</b> from both, and compare them for consistency.</p>
    """,
        unsafe_allow_html=True,
    )
with header_right:
    st.image("WESCO_International_logo.png", width=100)

# ---------- Upload Section ----------
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 Upload Journal Image")
    image_file = st.file_uploader("PNG only", type=["png"], key="image")
with col2:
    st.subheader("📄 Upload Journal Excel")
    excel_file = st.file_uploader("XLSX or XLS", type=["xlsx", "xls"], key="excel")

# ---------- Button Centered with Wesco Green ----------
left, center, right = st.columns([1, 4, 1])
with center:
    run = st.button("🔍 Extract and Compare", use_container_width=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #00A03E;
            color: white;
            font-size: 1.2em;
            padding: 0.8em;
            border-radius: 8px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

if run:
    # Preserve original guard/stop semantics
    if not image_file or not excel_file:
        st.warning("Please upload both image and Excel files.")
        st.stop()

    # Build and run the agent pipeline
    steps = [
        Step(name="prepare_image_state", run=prepare_image_state),
        Step(name="vision_extract_totals", run=vision_extract_totals),
        Step(name="extract_journal_table", run=extract_journal_table_step),
        Step(name="compute_excel_totals", run=compute_excel_totals_step),
    ]
    agent = Agent(steps)
    state: Dict[str, Any] = {
        "image_file": image_file,
        "excel_file": excel_file,
    }

    state = agent.execute(state)

    image_debit = state.get("image_debit")
    image_credit = state.get("image_credit")
    excel_debit = state.get("excel_debit")
    excel_credit = state.get("excel_credit")
    debit_col = state.get("debit_col")
    credit_col = state.get("credit_col")

    # Mirror original error message for vision parsing failure
    if state.get("vision_error"):
        st.error("❌ Could not extract debit/credit from image response.")

    # ---------- Comparison Result ----------
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center;'>
        <h2 style='color: #333;'>📌 Comparison Result</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if None not in (image_debit, image_credit, excel_debit, excel_credit):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🔎 Image Total Debit", value=f"${image_debit:,.2f}")
            st.metric(label="🔎 Image Total Credit", value=f"${image_credit:,.2f}")
        with col2:
            st.metric(label="📊 Excel Total Debit", value=f"${excel_debit:,.2f}")
            st.metric(label="📊 Excel Total Credit", value=f"${excel_credit:,.2f}")

        match = abs(image_debit - excel_debit) < 0.01 and abs(image_credit - excel_credit) < 0.01
        if match:
            st.markdown(
                """
            <div style='text-align: center; padding: 1em;'>
                <span style='font-size: 1.5em; color: green;'>✅ MATCH: Image and Excel totals are the same.</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style='text-align: center; padding: 1em;'>
                <span style='font-size: 1.5em; color: red;'>❌ NO MATCH: The extracted totals do not align.</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Unable to determine or compare totals from image and/or Excel.")

    # ---------- Preview Section ----------
    st.markdown("---")
    st.markdown("---")
    st.markdown("---")
    st.markdown("### 🔍 For Reference: Preview Extracted Data")

    # Columns for preview
    prev1, prev2 = st.columns(2)

    with prev1:
        st.markdown("🖼️ **Uploaded Image Preview**", unsafe_allow_html=True)
        img_bytes = state.get("image_bytes")
        if img_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            fixed_height = 400
            w_percent = (fixed_height / float(img.size[1]))
            width = int((float(img.size[0]) * float(w_percent)))
            img = img.resize((width, fixed_height))
            st.image(img, use_container_width=True)
        else:
            st.info("No image preview available.")

    with prev2:
        st.markdown("📊 **Uploaded Excel Preview**", unsafe_allow_html=True)
        df = state.get("df", pd.DataFrame())
        if debit_col and credit_col and not df.empty:
            st.dataframe(df[[debit_col, credit_col]].head(10), use_container_width=True, height=400)
        else:
            st.info("No preview available for Excel debit/credit columns.")

    # Surface any captured step errors without altering original UX flow
    if state.get("errors"):
        with st.expander("Debug: Agent step errors"):
            for e in state["errors"]:
                st.write(e)
