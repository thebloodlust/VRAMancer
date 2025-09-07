# dashboard/theme.py
"""
Theme helpers for the Dash/Streamlit dashboard.
"""

# --- Palette ------------------------------------------------------------
PRIMARY   = "#1f77b4"
SECONDARY = "#ff7f0e"
SUCCESS   = "#2ca02c"
DANGER    = "#d62728"
WARNING   = "#d8b365"
INFO      = "#17becf"

# --- Typography ---------------------------------------------------------
FONT_FAMILY = "'Inter', sans-serif"
FONT_SIZE   = "14px"

# --- Spacing -------------------------------------------------------------
PADDING = "12px"
MARGIN  = "12px"

# --- CSS generation ------------------------------------------------------
def get_css() -> str:
    """
    Return a minimal CSS string that can be dropped into a Dash/Streamlit
    page or written to a ``theme.css`` file.

    Example (Dash):

        import dashboard.theme as theme
        app = dash.Dash(__name__, external_stylesheets=[theme.get_css()])

    Example (Streamlit):

        import streamlit as st
        import dashboard.theme as theme
        st.markdown(f"<style>{theme.get_css()}</style>", unsafe_allow_html=True)
    """
    return f"""
    :root {{
        --color-primary:   {PRIMARY};
        --color-secondary: {SECONDARY};
        --color-success:   {SUCCESS};
        --color-danger:    {DANGER};
        --color-warning:   {WARNING};
        --color-info:      {INFO};

        --font-family: {FONT_FAMILY};
        --font-size:   {FONT_SIZE};

        --padding: {PADDING};
        --margin:   {MARGIN};
    }}

    body {{
        font-family: var(--font-family);
        font-size: var(--font-size);
        margin: 0;
        padding: 0;
        background: #fafafa;
    }}

    .card {{
        background: #fff;
        padding: var(--padding);
        margin: var(--margin);
        border: 1px solid #eaeaea;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,.1);
    }}

    .btn-primary {{
        background: var(--color-primary);
        color: #fff;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: background .15s ease;
    }}
    .btn-primary:hover {{ background: #115293; }}

    .btn-secondary {{
        background: var(--color-secondary);
        color: #fff;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        transition: background .15s ease;
    }}
    .btn-secondary:hover {{ background: #b86b00; }}
    """

# Optional helper -------------------------------------------------------
def color(name: str) -> str:
    """
    Return the hex code for a named color.
    """
    mapping = {
        "primary":   PRIMARY,
        "secondary": SECONDARY,
        "success":   SUCCESS,
        "danger":    DANGER,
        "warning":   WARNING,
        "info":      INFO,
    }
    return mapping.get(name.lower(), "#000000")
