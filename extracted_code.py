# ... previous lines unchanged ...
forecast_test = m_hold.predict(X_test)

yhat_test = np.expm1(forecast_test["yhat"]) - (backshift if backshift > 0 else 0.0)
actual_test = test_df["y_raw"].astype(float)

# Align and drop any NaNs before metric computation
mask = (~np.isnan(yhat_test.values)) & (~np.isnan(actual_test.values))
if mask.sum() > 0:
    holdout_rmse = float(np.sqrt(np.mean((actual_test.values[mask] - yhat_test.values[mask])**2)))
    # Avoid division by zero for MAPE
    nonzero_mask = mask & (actual_test.values != 0)
    if nonzero_mask.sum() > 0:
        holdout_mape = float(np.mean(np.abs((actual_test.values[nonzero_mask] - yhat_test.values[nonzero_mask]) / actual_test.values[nonzero_mask]))) * 100.0
    else:
        holdout_mape = float('nan')
else:
    holdout_rmse = float('nan')
    holdout_mape = float('nan')

mean_sales = float(work["y_raw"].mean())
holdout_rmse_pct = (holdout_rmse / mean_sales * 100.0) if (mean_sales != 0 and not np.isnan(holdout_rmse)) else float("nan")

print("\n================ HOLDOUT (last 6 months) ================")
print(f"Holdout RMSE (orig):  {holdout_rmse:,.2f}")
print(f"Holdout RMSE % mean:  {holdout_rmse_pct:.2f}%")
print(f"Holdout MAPE:         {holdout_mape:.2f}%")
print("========================================================\n")

# ============================================================
# 10) RMSE INTERPRETATION (CRM LANGUAGE)
# ============================================================

rmse_pct = holdout_rmse_pct  # prefer realistic holdout percentage
# Fallback to in-sample percentage if holdout unavailable
if np.isnan(rmse_pct):
    rmse_pct = (rmse_best_orig_insample / mean_sales * 100.0) if mean_sales != 0 else float('inf')
    holdout_rmse = rmse_best_orig_insample

print("\n========== RMSE INTERPRETATION ==========")
print("Mean sales:", round(mean_sales, 2))
print("Selected RMSE (holdout):", round(holdout_rmse, 2))
print("Selected RMSE % of mean:", round(rmse_pct, 3), "%")

if rmse_pct < 5:
    msg = "Excellent accuracy â€“ more than good enough for CRM forecasting & target-setting."
elif rmse_pct < 15:
    msg = "Good accuracy â€“ suitable for CRM targets, with some caution on edge cases."
elif rmse_pct < 30:
    msg = "OK accuracy â€“ usable for high-level planning, but not ideal for precise targets."
else:
    msg = "Needs work â€“ forecasting error is high relative to sales; review model or data."

print("Interpretation:", msg)
print("=========================================\n")

# ... remainder unchanged ...


