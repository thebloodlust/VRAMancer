# dashboard/theme.py
def get_css() -> str:
    """
    Return the URL of a simple CSS file hosted on a CDN.
    You can replace it by a local file if you want a custom theme.
    """
    return "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
