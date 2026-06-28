"""Utilities for injecting shared dashboard styling."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def load_css() -> None:
    """Inject the project CSS file into the active Streamlit page."""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
