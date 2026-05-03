# ============================================================
# Combined script:
#  - Optionally generate synthetic CRM sales data
#  - Read CSV (synthetic or real)
#  - Derive baseline from data
#  - Prophet baseline + tuned model
#  - RMSE interpretation + 12-month forecast
# ============================================================

import os, sys, subprocess, textwrap, time

# --- Auto-detect venv Python and re-launch if needed ---
_VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
_VENV_PYTHON = os.path.join(_VENV_DIR, "Scripts", "python.exe") if os.name == "nt" else os.path.join(_VENV_DIR, "bin", "python")
if os.path.isfile(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(_VENV_PYTHON):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

# --- Fix Windows console encoding for Unicode characters ---
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from datetime import datetime

# -----------------------------
# 0) SAFE NUMPY / PANDAS IMPORT
# -----------------------------
def safe_import_numpy_pandas():
    """
    Import numpy & pandas safely.
    If there's a binary mismatch, reinstall compatible versions.
    Add NumPy 2.x compatibility shims (np.float_, etc.).
    """
    global np, pd
    try:
        import numpy as _np
        import pandas as _pd
        np, pd = _np, _pd
    except Exception as e:
        print("Issue importing numpy/pandas:", e)
        print("Attempting to reinstall compatible numpy/pandas...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--force-reinstall",
                "numpy==1.26.4",
                "pandas==2.1.4",
            ],
            check=False,
        )
        import numpy as _np
        import pandas as _pd
        np, pd = _np, _pd
        print("✓ numpy/pandas reinstalled and imported successfully.")

    # NumPy 2.x compatibility shim
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    # If np.bool_ is missing, map it to built-in bool
    if not hasattr(np, "bool_"):
        np.bool_ = bool

safe_import_numpy_pandas()

# ============================================================
# 1) CONFIG: choose data source
# ============================================================

# True  → generate synthetic dataset first, then forecast on it
# False → skip generation and only read an existing CSV
USE_SYNTHETIC = True

# Path to CSV for forecasting
CSV_PATH = "territory_single_prophet_ready.csv"

# If USE_SYNTHETIC is False, set this to your real CRM file, e.g.:
# CSV_PATH = "my_real_crm_export.csv"


# ============================================================
# 2) (Optional) GENERATE SYNTHETIC SINGLE-TERRITORY DATA
# ============================================================
if USE_SYNTHETIC:
    print("\n=== GENERATING SYNTHETIC CRM DATA ===")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Date range: from 2022-01-31 to today's date (monthly)
    start_date = datetime(2022, 1, 31)
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    dates = pd.date_range(start=start_date, end=end_date, freq='M')

    # Single territory parameters (high performer, CRM scale ~414k)
    base_sales = 414_000
    base_territory_potential = 0.85
    base_close_rate = 0.32
    base_avg_deal = 45_000

    rows = []

    for idx, date in enumerate(dates):
        month_num = date.month
        year_num = date.year - start_date.year  # years since 2022

        # Trend: gradual growth over time (3–5% annually + slight monthly drift)
        trend_factor = 1 + (year_num * 0.04) + (idx * 0.003)

        # Seasonality: Q4 strongest, Q1 slower, summer slowdown, etc.
        if month_num in [11, 12]:        # Strong Q4
            seasonal_factor = 1.25
        elif month_num in [1, 2]:        # Slower Q1
            seasonal_factor = 0.90
        elif month_num in [6, 7, 8]:     # Summer slowdown
            seasonal_factor = 0.95
        elif month_num in [3, 4, 9, 10]: # Good months
            seasonal_factor = 1.08
        else:
            seasonal_factor = 1.0

        # Random variation (NARROWER: -2% to +2% instead of -10% to +10%)
        random_variation = np.random.uniform(0.98, 1.02)

        # Past sales with realistic pattern
        past_sales = int(base_sales * trend_factor * seasonal_factor * random_variation)

        # Open deals (correlated with past sales and deal size)
        base_deals = int((past_sales / base_avg_deal) * 1.5)
        open_deals = max(5, int(base_deals * np.random.uniform(0.95, 1.05)))

        # Close rate with slight variation, clipped between 15% and 50%
        close_rate = base_close_rate * np.random.uniform(0.97, 1.03)
        close_rate = min(0.50, max(0.15, close_rate))

        # Average deal size with mild variation
        avg_deal_size = int(base_avg_deal * np.random.uniform(0.95, 1.05))

        # Territory potential (relatively stable with very small variations)
        territory_potential = base_territory_potential * np.random.uniform(0.99, 1.01)
        territory_potential = min(1.0, max(0.5, territory_potential))

        # Target forecast based on past sales + optimistic growth (3–8% above)
        growth_expectation = np.random.uniform(1.03, 1.08)
        target_forecast = int(past_sales * growth_expectation * seasonal_factor)

        # High performer gets slightly higher targets
        target_forecast = int(target_forecast * 1.05)

        rows.append({
            "ds": date,
            "Target_Forecast": target_forecast,
            "past_sales": past_sales,
            "open_deals": open_deals,
            "avg_deal_size": avg_deal_size,
            "territory_potential": round(territory_potential, 3),
            "close_rate": round(close_rate, 3),
        })

    df_gen = pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)
    df_gen.to_csv(CSV_PATH, index=False)

    print("Synthetic data saved to:", os.path.abspath(CSV_PATH))
    print("Records:", len(df_gen))
    print("Date Range:", df_gen["ds"].min().date(), "to", df_gen["ds"].max().date())
    print("==========================================")


# ============================================================
# 3) READ DATASET (synthetic OR real)
# ============================================================

print("\n=== LOADING DATASET FOR FORECASTING ===")
try:
    df = pd.read_csv(CSV_PATH)
    if "ds" not in df.columns:
        raise ValueError("CSV must contain a 'ds' date column.")
    df["ds"] = pd.to_datetime(df["ds"])
    print(f"Data loaded. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"❌ CSV not found at {CSV_PATH}. Please adjust the path and rerun.")
    sys.exit(0)
except Exception as e:
    print("❌ Error reading dataset:", e)
    sys.exit(0)

# ============================================================
# 4) DERIVE BASE METRICS FROM DATA (first 12 months)
# ============================================================

df = df.sort_values("ds").reset_index(drop=True)

first_date = df["ds"].min()
base_period_end = first_date + pd.DateOffset(years=1)
base_df = df[df["ds"] <= base_period_end].copy()

if len(base_df) < 6:
    base_df = df.copy()

base_sales_ds = base_df["past_sales"].mean()
base_avg_deal_ds = base_df["avg_deal_size"].median()
base_close_rate_ds = base_df["close_rate"].median()
base_territory_potential_ds = base_df["territory_potential"].median()

print("\n=== Derived base parameters from dataset ===")
print("base_sales (from data):", round(base_sales_ds, 2))
print("base_avg_deal (from data):", round(base_avg_deal_ds, 2))
print("base_close_rate (from data):", round(base_close_rate_ds, 4))
print("base_territory_potential (from data):", round(base_territory_potential_ds, 3))
print("===========================================\n")


# ============================================================
# 5) PROPHET PIPELINE SETUP
# ============================================================

REQS = {
    "prophet": "1.1.5",
    "cmdstanpy": "1.2.4",
    "matplotlib": "3.8.4",
}

def sh(*args):
    print(">", " ".join(args))
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    return r.returncode == 0

def ver(pkg):
    try:
        import importlib.metadata as im
        return im.version(pkg)
    except Exception:
        try:
            import importlib_metadata as im
            return im.version(pkg)
        except Exception:
            return None

def ensure_no_local_shadow():
    p1 = os.path.abspath("prophet.py")
    p2 = os.path.abspath("prophet")
    if os.path.exists(p1) or os.path.isdir(p2):
        print(textwrap.dedent(f"""
        ❌ Found a local file/folder named 'prophet' that would shadow the library:
           {p1 if os.path.exists(p1) else p2}
        ➜ Please rename or remove it, then re-run this script.
        """).strip())
        return False
    return True

def pin_environment():
    sh(sys.executable, "-m", "pip", "uninstall", "-y", "fbprophet", "pystan")
    pkgs = []
    for k, v in REQS.items():
        cur = ver(k)
        if cur != v:
            pkgs.append(f"{k}=={v}")
    if pkgs:
        print("Pinning/aligning packages:", pkgs)
        sh(sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs)

def ensure_cmdstan(max_retries=1):
    try:
        import cmdstanpy as cs
        path = None
        try:
            path = cs.cmdstan_path()
        except Exception:
            path = None

        if not path or not os.path.isdir(path):
            print("CmdStan not found — installing (this can take several minutes)…")
            for attempt in range(max_retries + 1):
                try:
                    cs.install_cmdstan()
                    break
                except Exception as e:
                    print(f"CmdStan install attempt {attempt+1} failed: {e}")
                    if attempt < max_retries:
                        print("Retrying in 10s…")
                        time.sleep(10)
            try:
                path = cs.cmdstan_path()
            except Exception:
                path = None

        if path and os.path.isdir(path):
            os.environ["CMDSTAN"] = path
            print("CmdStan path:", path)
            return True, path

        msg = textwrap.dedent("""
        ❌ CmdStan is still not available.
        Possible fixes:
          • Ensure you have a C++ toolchain (gcc/clang) available.
          • In Colab, rerun this cell; cmdstanpy should build toolchain automatically.
        """).strip()
        print(msg)
        return False, msg

    except Exception as e:
        print("CmdStan check/import failed:", e)
        return False, str(e)

# ---------- bootstrap ----------
if not ensure_no_local_shadow():
    sys.exit(0)

pin_environment()

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import cmdstanpy as cs
    import prophet
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except Exception as e:
    print("Import error after pinning:", e)
    print("Tip: restart the kernel once, then run this script again.")
    sys.exit(0)

print("Versions →",
      f"numpy {np.__version__}, pandas {pd.__version__}, prophet {getattr(prophet,'__version__','?')},",
      f"cmdstanpy {getattr(cs,'__version__','?')}, matplotlib {matplotlib.__version__}")

ok, info = ensure_cmdstan(max_retries=1)
if not ok:
    print("Stopping before modeling because CmdStan is unavailable.\n")
    sys.exit(0)


# ============================================================
# 6) PREP DATA FOR PROPHET
# ============================================================

def iqr_outlier_mask(s, k=3.0):
    q1, q3 = np.percentile(s.dropna(), [25, 75])
    iqr = q3 - q1
    low, high = q1 - k*iqr, q3 + k*iqr
    return (s < low) | (s > high)

def fit_prophet(df_prop, regressors, params):
    m = Prophet(
        yearly_seasonality=params.get("yearly_seasonality", True),
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=params.get("seasonality_mode", "multiplicative"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 10.0),
        changepoint_range=params.get("changepoint_range", 0.8),
        n_changepoints=params.get("n_changepoints", 25),
        interval_width=params.get("interval_width", 0.8),
        stan_backend="CMDSTANPY",
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=params.get("monthly_fourier", 7))
    for r in regressors:
        m.add_regressor(r, standardize=True)
    m.fit(df_prop[["ds", "y"] + regressors])
    return m

def quick_cv_rmse(m, horizon_days=365, period_days=180, initial_frac=0.6):
    history = m.history["ds"].max() - m.history["ds"].min()
    history_days = history.days if hasattr(history, "days") else int(history / np.timedelta64(1, 'D'))
    initial_days = max(365, int(history_days * initial_frac))
    df_cv = cross_validation(
        m,
        horizon=f"{horizon_days} days",
        period=f"{period_days} days",
        initial=f"{initial_days} days",
        parallel="processes",
    )
    pm = performance_metrics(df_cv, rolling_window=1)
    return float(pm["rmse"].iloc[-1])

required_columns = ['past_sales', 'open_deals', 'avg_deal_size', 'territory_potential', 'close_rate']
missing = [c for c in required_columns if c not in df.columns]
if missing:
    print(f"❌ Missing required columns: {missing}")
    sys.exit(0)

target_column = next((c for c in ['y', 'past_sales', 'actual_sales', 'sales', 'revenue', 'target']
                      if c in df.columns), None)
if not target_column:
    print("❌ No target variable column found. Expected one of: y, past_sales, actual_sales, sales, revenue, target")
    sys.exit(0)

work = df.copy()
work["ds"] = pd.to_datetime(work["ds"])
work["y_raw"] = work[target_column].astype(float)

work = work.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)
work[required_columns] = work[required_columns].ffill().bfill()

# Outlier capping
out_mask = iqr_outlier_mask(work["y_raw"], k=3.0)
if out_mask.any():
    print(f"Outliers detected: {int(out_mask.sum())} rows. Capping to IQR bounds.")
    y = work["y_raw"].copy()
    q1, q3 = np.percentile(y.dropna(), [25, 75])
    iqr = q3 - q1
    low, high = q1 - 3.0*iqr, q3 + 3.0*iqr
    work["y_raw"] = y.clip(lower=low, upper=high)

# Variance stabilization (log1p)
if (work["y_raw"] <= 0).any():
    shift = abs(work["y_raw"].min()) + 1.0
    print(f"Non-positive values found; shifting target by +{shift:.2f} before log1p.")
    work["y_shift"] = work["y_raw"] + shift
    work["y"] = np.log1p(work["y_shift"])
    backshift = shift
else:
    work["y"] = np.log1p(work["y_raw"])
    backshift = 0.0

regressors = required_columns


# ============================================================
# 7) BASELINE MODEL
# ============================================================

print("\nTraining baseline Prophet model...")
baseline_params = dict(
    yearly_seasonality=True,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    changepoint_range=0.8,
    n_changepoints=25,
    monthly_fourier=7,
    interval_width=0.8,
)
m_base = fit_prophet(work, regressors, baseline_params)
rmse_base_cv_log = quick_cv_rmse(m_base)

print(f"Baseline CV RMSE (log-space): {rmse_base_cv_log:.4f} (lower is better)")
pred_hist_base = m_base.predict(work[["ds"] + regressors])
yhat_base = np.expm1(pred_hist_base["yhat"])
if backshift > 0:
    yhat_base = yhat_base - backshift
rmse_base_orig = float(np.sqrt(np.mean((work["y_raw"] - yhat_base)**2)))
print(f"Baseline In-sample RMSE (original scale): {rmse_base_orig:,.2f}")


# ============================================================
# 8) TUNING – BY RMSE (ORIGINAL SCALE)
# ============================================================

print("\nTuning hyperparameters to reduce RMSE (original scale)...")
grid = [
    {"changepoint_prior_scale": 0.03, "seasonality_prior_scale": 5.0,  "seasonality_mode": "multiplicative", "monthly_fourier": 7, "n_changepoints": 25},
    {"changepoint_prior_scale": 0.03, "seasonality_prior_scale": 15.0, "seasonality_mode": "multiplicative", "monthly_fourier": 9, "n_changepoints": 35},
    {"changepoint_prior_scale": 0.10, "seasonality_prior_scale": 5.0,  "seasonality_mode": "multiplicative", "monthly_fourier": 7, "n_changepoints": 25},
    {"changepoint_prior_scale": 0.10, "seasonality_prior_scale": 15.0, "seasonality_mode": "multiplicative", "monthly_fourier": 9, "n_changepoints": 35},
    {"changepoint_prior_scale": 0.20, "seasonality_prior_scale": 10.0, "seasonality_mode": "multiplicative", "monthly_fourier": 12, "n_changepoints": 50},
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 20.0, "seasonality_mode": "multiplicative", "monthly_fourier": 12, "n_changepoints": 25},
    {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0, "seasonality_mode": "additive",       "monthly_fourier": 7, "n_changepoints": 25},
    {"changepoint_prior_scale": 0.15, "seasonality_prior_scale": 5.0,  "seasonality_mode": "additive",       "monthly_fourier": 9, "n_changepoints": 35},
]

best_model = m_base
best_params = baseline_params.copy()
best_rmse_orig = rmse_base_orig

for i, p in enumerate(grid, 1):
    params = baseline_params.copy()
    params.update(p)

    m_try = fit_prophet(work, regressors, params)
    pred_try = m_try.predict(work[["ds"] + regressors])
    yhat_try = np.expm1(pred_try["yhat"])
    if backshift > 0:
        yhat_try = yhat_try - backshift
    rmse_try_orig = float(np.sqrt(np.mean((work["y_raw"] - yhat_try)**2)))

    print(f"  Try {i}/{len(grid)} -> RMSE (original): {rmse_try_orig:,.4f}, params: {p}")

    if rmse_try_orig < best_rmse_orig:
        best_rmse_orig = rmse_try_orig
        best_model = m_try
        best_params = params

best_cv_rmse_log = quick_cv_rmse(best_model)

pred_hist_best = best_model.predict(work[["ds"] + regressors])
yhat_best = np.expm1(pred_hist_best["yhat"])
if backshift > 0:
    yhat_best = yhat_best - backshift
rmse_best_orig = float(np.sqrt(np.mean((work["y_raw"] - yhat_best)**2)))

print("\n================ MODEL SELECTION ================")
print(f"Baseline CV RMSE (log):    {rmse_base_cv_log:.4f}")
print(f"Tuned   CV RMSE (log):     {best_cv_rmse_log:.4f}")
print(f"Baseline RMSE (original):  {rmse_base_orig:,.2f}")
print(f"Tuned   RMSE (original):   {rmse_best_orig:,.2f}")
print("Chosen params:", best_params)


# ============================================================
# 9) RMSE INTERPRETATION (CRM LANGUAGE)
# ============================================================

mean_sales = float(work["y_raw"].mean())
rmse_pct = rmse_best_orig / mean_sales * 100 if mean_sales != 0 else float("inf")

print("\n========== RMSE INTERPRETATION ==========")
print("Mean sales:", round(mean_sales, 2))
print("Tuned RMSE:", round(rmse_best_orig, 2))
print("Tuned RMSE % of mean:", round(rmse_pct, 3), "%")

if rmse_pct < 5:
    msg = "Excellent accuracy – more than good enough for CRM forecasting & target-setting."
elif rmse_pct < 15:
    msg = "Good accuracy – suitable for CRM targets, with some caution on edge cases."
elif rmse_pct < 30:
    msg = "OK accuracy – usable for high-level planning, but not ideal for precise targets."
else:
    msg = "Needs work – forecasting error is high relative to sales; review model or data."

print("Interpretation:", msg)
print("=========================================\n")


# ============================================================
# 10) FUTURE FRAME (12 MONTHS) + REGRESSOR PROJECTIONS
# ============================================================

last_date = work["ds"].max()
future_start = last_date + pd.DateOffset(months=1)
future_dates = pd.date_range(start=future_start, periods=12, freq="MS")
future_df = pd.DataFrame({"ds": future_dates})

print("\nProjecting future regressor values…")
for reg in regressors:
    recent = work[reg].tail(6)
    if len(recent) >= 2:
        x = np.arange(len(recent))
        z = np.polyfit(x, recent, 1)
        trend = z[0]; last_val = recent.iloc[-1]
        sigma = work[reg].std()
        vals = []
        for i in range(12):
            proj = last_val + trend * (i + 1)
            noise = np.random.normal(0, max(1e-6, sigma) * 0.02)
            vals.append(max(0, proj + noise))
        future_df[reg] = vals
    else:
        future_df[reg] = recent.iloc[-1] if len(recent) else work[reg].median()


# ============================================================
# 11) FORECAST & BACK-TRANSFORM
# ============================================================

print("\nGenerating forecasts with tuned model…")
forecast = best_model.predict(future_df)
yhat_future = np.expm1(forecast["yhat"])
yhat_lower = np.expm1(forecast["yhat_lower"])
yhat_upper = np.expm1(forecast["yhat_upper"])
if backshift > 0:
    yhat_future -= backshift; yhat_lower -= backshift; yhat_upper -= backshift

forecast_summary = pd.DataFrame({
    "Month": future_df["ds"].dt.strftime("%Y-%m"),
    "Forecast": yhat_future.round(0),
    "Lower_Bound": yhat_lower.round(0),
    "Upper_Bound": yhat_upper.round(0),
})
forecast_summary["Target_Sales"] = (forecast_summary["Forecast"] * 1.10).round(0)

print("\n================ 12-MONTH SALES FORECAST ================")
print(forecast_summary)


# ============================================================
# 12) VISUALS
# ============================================================

print("\nPlotting…")
complete_future = pd.concat([work[["ds"] + regressors], future_df], ignore_index=True)
complete_pred = best_model.predict(complete_future)
comp_yhat = np.expm1(complete_pred["yhat"])
comp_low = np.expm1(complete_pred["yhat_lower"])
comp_up  = np.expm1(complete_pred["yhat_upper"])
if backshift > 0:
    comp_yhat -= backshift; comp_low -= backshift; comp_up -= backshift

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
ax1.plot(work["ds"], work["y_raw"], "o-", label="Historical Sales", linewidth=2, markersize=4)
ax1.plot(complete_pred["ds"], comp_yhat, "--", label="Prophet Forecast", linewidth=2, alpha=0.9)
ax1.fill_between(complete_pred["ds"], comp_low, comp_up, alpha=0.25, label="Confidence Interval")
ax1.axvline(x=work["ds"].max(), color="red", linestyle=":", alpha=0.7, label="Forecast Start")
ax1.set_title("Sales Forecast (Historical + 12-Month Projection)")
ax1.set_ylabel("Sales"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(future_df["ds"], yhat_future, "o-", label="12-Month Forecast", linewidth=2, markersize=6)
ax2.plot(future_df["ds"], yhat_future * 1.10, "^--", label="Target (+10%)", linewidth=2, markersize=5)
ax2.fill_between(future_df["ds"], yhat_lower, yhat_upper, alpha=0.25, label="Confidence Interval")
ax2.set_title("12-Month Forecast Detail")
ax2.set_ylabel("Sales"); ax2.set_xlabel("Date"); ax2.legend(); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

fig2 = best_model.plot_components(complete_pred)
plt.suptitle("Prophet Components (internal scale)")
plt.tight_layout(); plt.show()


# ============================================================
# 13) EXPORT RESULTS
# ============================================================

results_df = pd.DataFrame({
    'Date': future_df['ds'],
    'Forecast_Sales': yhat_future.round(0),
    'Target_Sales': (yhat_future * 1.10).round(0),
    'Lower_Bound': yhat_lower.round(0),
    'Upper_Bound': yhat_upper.round(0),
    'Past_Sales_Input': future_df['past_sales'].round(0),
    'Open_Deals_Input': future_df['open_deals'].round(0),
    'Avg_Deal_Size_Input': future_df['avg_deal_size'].round(0),
    'Territory_Potential_Input': future_df['territory_potential'].round(3),
    'Close_Rate_Input': future_df['close_rate'].round(2)
})
results_df.to_csv('prophet_sales_forecast_results.csv', index=False)
print("✓ Forecast results exported to 'prophet_sales_forecast_results.csv'")

historical_performance = pd.DataFrame({
    'Date': work['ds'],
    'Actual_Sales': work['y_raw'],
    'Predicted_Sales': yhat_best,
    'Error': work['y_raw'] - yhat_best,
    'Absolute_Error': np.abs(work['y_raw'] - yhat_best),
    'Percentage_Error': ((work['y_raw'] - yhat_best) / work['y_raw'] * 100).replace([np.inf, -np.inf], np.nan)
})
historical_performance.to_csv('prophet_historical_performance.csv', index=False)
print("✓ Historical performance exported to 'prophet_historical_performance.csv'")

print("\n================ SUMMARY ================")
print(f"✓ Tuned model chosen (by lower RMSE on original scale).")
print(f"✓ 12-month forecast starts {future_dates[0].date()}")
print(f"✓ Avg monthly forecast: {yhat_future.mean():,.0f}")
print(f"✓ Total 12-month forecast: {yhat_future.sum():,.0f}")
print("=========================================")
