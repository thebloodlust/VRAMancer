# dashboard/app.py
import dash
from dash import html, dcc, dash_table
import plotly.express as px
from core import GPUMonitor
from .theme import get_css

# ---------- Dash app ----------
app = dash.Dash(
    __name__,
    external_stylesheets=[get_css()],
    title="GPU Monitor – VR‑C",
)

# ---------- Helpers ----------
def _mb(x: int) -> str:
    return f"{x // 1024**2:,} MB"

# ---------- Layout ----------
app.layout = html.Div(
    [
        html.H1("GPU Monitor", className="card"),
        dash_table.DataTable(
            id="gpus",
            columns=[
                {"name": "Index", "id": "index"},
                {"name": "Name", "id": "name"},
                {"name": "Type", "id": "type"},
                {"name": "Total MB", "id": "total_memory"},
                {"name": "Allocated MB", "id": "allocated"},
            ],
            data=[{
                "index": g["index"],
                "name": g["name"],
                "type": g["type"],
                "total_memory": f"{g['total_memory']//1024**2:,} MB" if g["total_memory"] else "N/A",
                "allocated": _mb(GPUMonitor().memory_allocated(g["index"])),
            } for g in GPUMonitor().gpus],
            style_cell={"textAlign": "left"},
        ),
        dcc.Interval(id="refresh", interval=2000, n_intervals=0),
    ],
    title="GPU Monitor – VR‑C",
)

# ---------- Callbacks ----------
@app.callback(
    dash.dependencies.Output("gpus", "data"),
    [dash.dependencies.Input("refresh", "n_intervals")],
)
def refresh_gpus(_):
    monitor = GPUMonitor()
    return [
        {
            "index": g["index"],
            "name": g["name"],
            "type": g["type"],
            "total_memory": f"{g['total_memory']//1024**2:,} MB" if g["total_memory"] else "N/A",
            "allocated": _mb(monitor.memory_allocated(g["index"])),
        }
        for g in monitor.gpus
    ]

# ---------- Run ----------
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
