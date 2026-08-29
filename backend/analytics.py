"""Port of the energy baseline regression models from utils.py.

Operates on pandas DataFrames constructed from DB query results instead of CSVs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# SEU category → normalization method
SEU_NORMS = {
    "Boiler Systems (Gas)": "hdd",
    "Air Handling Units (Gas)": "hdd",
    "Catering Equipment": "opday",
    "Lighting Systems": "opday",
    "Air Conditioning & Refrigeration": "cdd",
    "Electric Space Heaters": "hdd",
    "ICT & Server Room Cooling": "fixed",
    "EV Charging Infrastructure": "opday",
    "Onsite Solar PV": "pv",
}


def build_dataframe(readings, weather, seu_category, utility_type):
    """Build a daily DataFrame for a meter from DB rows."""
    if not readings:
        return pd.DataFrame()

    df = pd.DataFrame(readings)
    df["date"] = pd.to_datetime(df["date"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce")
    df = df.dropna(subset=["date", "consumption"])

    if weather:
        wdf = pd.DataFrame(weather)
        wdf["date"] = pd.to_datetime(wdf["date"])
        for col in ("hdd", "cdd"):
            if col in wdf.columns:
                wdf[col] = pd.to_numeric(wdf[col], errors="coerce")
        df = df.merge(wdf[["date", "hdd", "cdd"]], on="date", how="left")
    else:
        df["hdd"] = np.nan
        df["cdd"] = np.nan

    df["year"] = df["date"].dt.year
    df["is_operational"] = df["consumption"] > 0
    df["seu_category"] = seu_category
    df["utility_type"] = utility_type
    return df


def evaluate_meter_models(data: pd.DataFrame, train_year: int, test_year: int):
    """Run regression per meter, return summary DataFrame.

    Mirrors the original evaluate_meter_models from utils.py.
    """
    results = []
    data = data.copy()
    if "is_operational" not in data.columns:
        data["is_operational"] = data["consumption"] > 0
    if "seu_category" not in data.columns:
        data["seu_category"] = "Unknown"
    data["seu_category"] = data["seu_category"].fillna("Unknown")

    for meter in data["meter"].unique():
        meter_data = data[data["meter"] == meter].copy()
        if meter_data.empty:
            continue
        seu = meter_data["seu_category"].iloc[0] if pd.notna(meter_data["seu_category"].iloc[0]) else "Unknown"
        norm = SEU_NORMS.get(seu, "opday")
        train = meter_data[meter_data["year"] == train_year]
        test = meter_data[meter_data["year"] == test_year]
        if len(train) < 10 or len(test) < 10:
            continue
        # Align train to same date range as test
        if not test.empty:
            max_test_date = test["date"].max()
            baseline_cutoff = pd.Timestamp(year=train_year, month=max_test_date.month, day=max_test_date.day)
            train = train[train["date"] <= baseline_cutoff]

        if norm in ("hdd", "cdd"):
            climate_col = "hdd" if norm == "hdd" else "cdd"
            train = train.dropna(subset=[climate_col, "consumption"])
            test = test.dropna(subset=[climate_col, "consumption"])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[[climate_col, "is_operational"]]
            y_train = train["consumption"]
            X_test = test[[climate_col, "is_operational"]]
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif norm in ("fixed", "pv"):
            y_pred = np.zeros(len(test))
        else:  # opday
            train = train.dropna(subset=["consumption"])
            test = test.dropna(subset=["consumption"])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[["is_operational"]]
            y_train = train["consumption"]
            X_test = test[["is_operational"]]
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)

        predicted = float(np.sum(y_pred)) if norm not in ("fixed", "pv") else float(test["consumption"].sum())
        actual = float(test["consumption"].sum())
        results.append({
            "meter": str(meter),
            "seu_category": seu,
            "baseline": float(train["consumption"].sum()),
            "predicted": round(predicted, 1),
            "actual": round(actual, 1),
            "estimated_savings": round(predicted - actual, 0),
            "baseline_days": int(train["is_operational"].sum()),
            "actual_days": int(test["is_operational"].sum()),
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df["pct_savings"] = np.where(df["actual"] != 0, round(100 * df["estimated_savings"] / df["actual"], 1), None)
    return df


def get_monthly_comparison(data: pd.DataFrame, summary_df: pd.DataFrame, climate_col: str, train_year: int, test_year: int, max_meters: int = 8):
    """Return monthly actual vs predicted data for charting."""
    full_months = pd.Index(range(1, 13), name="month")
    summary_df = summary_df.sort_values(by="actual", ascending=False)
    meters = summary_df["meter"].tolist()[:max_meters]
    charts = []

    for meter in meters:
        df_train = data[(data["meter"] == meter) & (data["year"] == train_year)].copy()
        df_test = data[(data["meter"] == meter) & (data["year"] == test_year)].copy()
        if climate_col in df_train.columns:
            df_train = df_train.dropna(subset=[climate_col, "consumption"])
            df_test = df_test.dropna(subset=[climate_col, "consumption"])
        if df_train.empty or df_test.empty:
            continue
        X_cols = [c for c in [climate_col, "is_operational"] if c in df_train.columns]
        model = LinearRegression().fit(df_train[X_cols], df_train["consumption"])
        df_test = df_test.copy()
        df_test["predicted"] = model.predict(df_test[X_cols])

        monthly_train = df_train.groupby(df_train["date"].dt.month)["consumption"].sum().reindex(full_months, fill_value=0)
        monthly_test = df_test.groupby(df_test["date"].dt.month)["consumption"].sum().reindex(full_months, fill_value=0)
        monthly_pred = df_test.groupby(df_test["date"].dt.month)["predicted"].sum().reindex(full_months, fill_value=0)

        charts.append({
            "meter": meter,
            "months": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            "baseline": monthly_train.values.tolist(),
            "actual": monthly_test.values.tolist(),
            "predicted": [float(v) for v in monthly_pred.values],
        })
    return charts


def build_sankey(gas_summary: pd.DataFrame, elec_summary: pd.DataFrame, total_gas: float, total_elec: float):
    """Build Sankey diagram data for SEU energy flow."""
    labels = ["Total Energy", "Gas", "Electricity"]
    sources, targets, values = [], [], []

    if total_gas > 0:
        sources.append(0); targets.append(1); values.append(float(total_gas))
    if total_elec > 0:
        sources.append(0); targets.append(2); values.append(float(total_elec))

    gas_seu = gas_summary.groupby("seu_category")["actual"].sum() if not gas_summary.empty else pd.Series(dtype=float)
    elec_seu = elec_summary.groupby("seu_category")["actual"].sum() if not elec_summary.empty else pd.Series(dtype=float)
    gas_seu = gas_seu[gas_seu > 0]
    elec_seu = elec_seu[elec_seu > 0]

    idx = 3
    for seu, val in gas_seu.items():
        labels.append(f"{seu}")
        sources.append(1); targets.append(idx); values.append(float(val))
        idx += 1
    for seu, val in elec_seu.items():
        labels.append(f"{seu}")
        sources.append(2); targets.append(idx); values.append(float(val))
        idx += 1

    return {"labels": labels, "sources": sources, "targets": targets, "values": values}
