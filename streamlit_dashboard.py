import io
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
FORECAST_SCRIPT = BASE_DIR / "forecast_backup.py"
HISTORICAL_CSV = BASE_DIR / "prophet_historical_performance.csv"
FORECAST_CSV = BASE_DIR / "prophet_sales_forecast_results.csv"
README_PATH = BASE_DIR / "README.md"


st.set_page_config(page_title="CRM Sales Forecast & Target Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .app-card {
        border: 1px solid #d9e3f0;
        border-radius: 14px;
        padding: 18px;
        background: #f8fbff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_forecast_script() -> tuple[bool, str]:
    if not FORECAST_SCRIPT.exists():
        return False, f"Missing script: {FORECAST_SCRIPT.name}"

    cmd = [sys.executable, str(FORECAST_SCRIPT)]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        return False, output.strip()
    return True, output.strip()


def load_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    hist_df = pd.read_csv(HISTORICAL_CSV) if HISTORICAL_CSV.exists() else None
    forecast_df = pd.read_csv(FORECAST_CSV) if FORECAST_CSV.exists() else None
    return hist_df, forecast_df


def calculate_rmse(hist_df: pd.DataFrame) -> float:
    if "Error" in hist_df.columns:
        return float(np.sqrt(np.mean(np.square(hist_df["Error"]))))
    if {"Actual_Sales", "Predicted_Sales"}.issubset(hist_df.columns):
        diff = hist_df["Actual_Sales"] - hist_df["Predicted_Sales"]
        return float(np.sqrt(np.mean(np.square(diff))))
    return float("nan")


def calculate_mape(hist_df: pd.DataFrame) -> float:
    if "Percentage_Error" in hist_df.columns:
        return float(hist_df["Percentage_Error"].abs().mean())
    if {"Actual_Sales", "Predicted_Sales"}.issubset(hist_df.columns):
        actual = hist_df["Actual_Sales"].replace(0, np.nan)
        ape = (hist_df["Actual_Sales"] - hist_df["Predicted_Sales"]).abs() / actual
        return float((ape.mean()) * 100)
    return float("nan")


def calculate_rmse_pct(hist_df: pd.DataFrame, rmse: float) -> float:
    if not np.isfinite(rmse):
        return float("nan")
    if "Actual_Sales" not in hist_df.columns:
        return float("nan")
    mean_actual = float(hist_df["Actual_Sales"].mean())
    if mean_actual == 0:
        return float("nan")
    return float((rmse / mean_actual) * 100)


def draw_historical_chart(hist_df: pd.DataFrame) -> None:
    if not {"Actual_Sales", "Predicted_Sales"}.issubset(hist_df.columns):
        st.warning("Historical CSV is missing required columns for charting.")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist_df["Actual_Sales"].values, label="Actual", linewidth=2)
    ax.plot(hist_df["Predicted_Sales"].values, label="Predicted", linewidth=2)
    ax.set_title("Historical: Actual vs Predicted")
    ax.set_xlabel("Row Index")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def draw_forecast_chart(forecast_df: pd.DataFrame) -> None:
    required = {"Month", "Forecast_Sales"}
    if not required.issubset(forecast_df.columns):
        st.warning("Forecast CSV is missing required columns for charting.")
        return

    chart_df = forecast_df.copy()
    chart_df["Month"] = pd.to_datetime(chart_df["Month"], errors="coerce")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(chart_df["Month"], chart_df["Forecast_Sales"], marker="o", linewidth=2, label="Forecast")

    if "Target_Sales" in chart_df.columns:
        ax.plot(chart_df["Month"], chart_df["Target_Sales"], marker="s", linewidth=2, label="Target")

    if {"Lower_Bound", "Upper_Bound"}.issubset(chart_df.columns):
        ax.fill_between(
            chart_df["Month"],
            chart_df["Lower_Bound"],
            chart_df["Upper_Bound"],
            alpha=0.15,
            label="Confidence Band",
        )

    ax.set_title("Future Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def read_readme() -> str:
    if not README_PATH.exists():
        return ""
    return README_PATH.read_text(encoding="utf-8", errors="replace")


if "last_run_ok" not in st.session_state:
    st.session_state.last_run_ok = False
if "last_run_log" not in st.session_state:
    st.session_state.last_run_log = ""
if "show_readme" not in st.session_state:
    st.session_state.show_readme = False


st.title("CRM Sales Forecast & Target Dashboard")
st.caption("Clean Streamlit dashboard for running forecast, viewing RMSE and charts, and downloading CSV files.")

st.markdown('<div class="app-card">', unsafe_allow_html=True)

run_clicked = st.button("Run Forecast", type="primary", use_container_width=True)
if run_clicked:
    with st.spinner("Running forecast script..."):
        ok, log = run_forecast_script()
        st.session_state.last_run_ok = ok
        st.session_state.last_run_log = log
        st.session_state.show_readme = False

if st.session_state.last_run_log:
    with st.expander("Run Log", expanded=not st.session_state.last_run_ok):
        st.text(st.session_state.last_run_log)

if st.session_state.last_run_ok:
    open_readme = st.button("Open Generated README", use_container_width=True)
    if open_readme:
        st.session_state.show_readme = True

if st.session_state.show_readme:
    readme_text = read_readme()
    if readme_text:
        with st.expander("README Preview", expanded=True):
            st.markdown(readme_text)
    else:
        st.warning("README.md was not found.")

hist_df, forecast_df = load_results()

if hist_df is None and forecast_df is None:
    st.info("No result files found yet. Click 'Run Forecast' to generate outputs.")
else:
    st.subheader("KPI")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    rmse = float("nan")
    rmse_pct = float("nan")
    mape = float("nan")
    if hist_df is not None:
        rmse = calculate_rmse(hist_df)
        rmse_pct = calculate_rmse_pct(hist_df, rmse)
        mape = calculate_mape(hist_df)

    with metric_col1:
        st.metric("RMSE", f"{rmse:,.2f}" if np.isfinite(rmse) else "N/A")
    with metric_col2:
        st.metric("RMSE%", f"{rmse_pct:,.2f}%" if np.isfinite(rmse_pct) else "N/A")
    with metric_col3:
        st.metric("MAPE", f"{mape:,.2f}%" if np.isfinite(mape) else "N/A")

    left, right = st.columns(2)

    with left:
        if hist_df is not None:
            draw_historical_chart(hist_df)
        else:
            st.warning("Historical results file not found.")

    with right:
        if forecast_df is not None:
            draw_forecast_chart(forecast_df)
        else:
            st.warning("Forecast results file not found.")

    st.subheader("Download CSV")
    dl1, dl2 = st.columns(2)
    with dl1:
        if forecast_df is not None:
            st.download_button(
                label="Download Forecast CSV",
                data=csv_bytes(forecast_df),
                file_name="prophet_sales_forecast_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with dl2:
        if hist_df is not None:
            st.download_button(
                label="Download Historical CSV",
                data=csv_bytes(hist_df),
                file_name="prophet_historical_performance.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.markdown("</div>", unsafe_allow_html=True)
