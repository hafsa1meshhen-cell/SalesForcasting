import io
import re
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
GENERATED_DATASET_CSV = BASE_DIR / "territory_single_prophet_ready.csv"
README_PATH = BASE_DIR / "README.md"


st.set_page_config(page_title="CRM Sales Forecast & Target Dashboard", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #1e293b;
        --ocean: #0f766e;
        --sunset: #ef4444;
        --sky: #0ea5e9;
        --sand: #fff7ed;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.2) 0%, rgba(14, 165, 233, 0) 40%),
            radial-gradient(circle at 85% 10%, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0) 35%),
            linear-gradient(145deg, #f8fafc 0%, #eef6ff 52%, #fff7ed 100%);
        color: var(--ink);
    }

    .main .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.62);
        border: 1px solid rgba(14, 165, 233, 0.18);
        border-radius: 20px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(6px);
    }

    h1, h2, h3 {
        color: #0b3c5d;
        letter-spacing: 0.02em;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 14px;
        padding: 8px 12px;
        box-shadow: 0 8px 20px rgba(14, 165, 233, 0.1);
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--sunset), #f97316);
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 0.45rem 1.35rem;
        font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 8px 18px rgba(239, 68, 68, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(239, 68, 68, 0.32);
    }

    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, var(--ocean), var(--sky));
        color: #ffffff;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(15, 118, 110, 0.24);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(15, 118, 110, 0.32);
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(14, 165, 233, 0.25);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.75);
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


def load_holdout_metrics_from_readme() -> tuple[float, float, float]:
    text = read_readme()
    if not text:
        return float("nan"), float("nan"), float("nan")

    rmse = float("nan")
    rmse_pct = float("nan")
    mape = float("nan")

    rmse_match = re.search(
        r"\*\*Holdout RMSE\*\*:\s*\*\*([\d,]+(?:\.\d+)?)\*\*\s*\(\*\*([\d.]+)%\*\*",
        text,
    )
    if rmse_match:
        rmse = float(rmse_match.group(1).replace(",", ""))
        rmse_pct = float(rmse_match.group(2))

    mape_match = re.search(r"\*\*MAPE\*\*:\s*\*\*([\d.]+)%\*\*", text)
    if mape_match:
        mape = float(mape_match.group(1))

    return rmse, rmse_pct, mape


if "last_run_ok" not in st.session_state:
    st.session_state.last_run_ok = False
if "last_run_log" not in st.session_state:
    st.session_state.last_run_log = ""
if "show_readme" not in st.session_state:
    st.session_state.show_readme = False
if "result_hist_df" not in st.session_state or "result_forecast_df" not in st.session_state:
    initial_hist, initial_forecast = load_results()
    st.session_state.result_hist_df = initial_hist
    st.session_state.result_forecast_df = initial_forecast


st.title("CRM Sales Forecast & Target Dashboard")
st.markdown("**Dynamic dashboard for running forecast, viewing RMSE and charts, and downloading CSV files**")

status_placeholder = st.empty()
log_placeholder = st.empty()
tabs_placeholder = st.empty()

action_col1, action_col2 = st.columns([1, 1])
with action_col1:
    run_clicked = st.button("Run Forecast", type="primary", use_container_width=True)
with action_col2:
    reset_clicked = st.button("Reset Results", use_container_width=True)

if reset_clicked:
    st.session_state.last_run_ok = False
    st.session_state.last_run_log = ""
    st.session_state.show_readme = False
    st.session_state.result_hist_df = None
    st.session_state.result_forecast_df = None
    status_placeholder.success("Previous forecast results cleared.")

if run_clicked:
    status_placeholder.info("Running forecast...")
    with st.spinner("Running forecast script..."):
        ok, log = run_forecast_script()
        st.session_state.last_run_ok = ok
        st.session_state.last_run_log = log
        st.session_state.show_readme = False
        if ok:
            new_hist_df, new_forecast_df = load_results()
            st.session_state.result_hist_df = new_hist_df
            st.session_state.result_forecast_df = new_forecast_df
            status_placeholder.success("Forecast run completed.")
        else:
            status_placeholder.error("Forecast run failed. Check Run Log for details.")

if st.session_state.last_run_log:
    with log_placeholder.container():
        with st.expander("Run Log", expanded=not st.session_state.last_run_ok):
            st.text(st.session_state.last_run_log)
else:
    log_placeholder.empty()

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

hist_df = st.session_state.result_hist_df
forecast_df = st.session_state.result_forecast_df

with tabs_placeholder.container():
    downloads_tab, metrics_tab = st.tabs(["Download CSV", "KPT Metrics & Charts"])

    with downloads_tab:
        st.subheader("Download CSV")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            if forecast_df is not None:
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv_bytes(forecast_df),
                    file_name="prophet_sales_forecast_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No forecast CSV available.")
        with dl2:
            if hist_df is not None:
                st.download_button(
                    label="Download Historical CSV",
                    data=csv_bytes(hist_df),
                    file_name="prophet_historical_performance.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No historical CSV available.")
        with dl3:
            if forecast_df is not None and GENERATED_DATASET_CSV.exists():
                st.download_button(
                    label="Download generated dataset",
                    data=GENERATED_DATASET_CSV.read_bytes(),
                    file_name="territory_single_prophet_ready.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No generated dataset available.")

    with metrics_tab:
        st.subheader("KPI metrics")
        if hist_df is None and forecast_df is None:
            st.info("No result files found yet. Click 'Run Forecast' to generate outputs.")
        else:
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            holdout_rmse, holdout_rmse_pct, holdout_mape = load_holdout_metrics_from_readme()

            rmse = holdout_rmse
            rmse_pct = holdout_rmse_pct
            mape = holdout_mape

            if hist_df is not None:
                if not np.isfinite(rmse):
                    rmse = calculate_rmse(hist_df)
                if not np.isfinite(rmse_pct):
                    rmse_pct = calculate_rmse_pct(hist_df, rmse)
                if not np.isfinite(mape):
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

            with st.expander("Forecast Table", expanded=False):
                if forecast_df is not None:
                    st.dataframe(forecast_df, use_container_width=True)
                else:
                    st.info("No forecast table available.")

