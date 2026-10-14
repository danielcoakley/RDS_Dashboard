"""Port of the energy baseline regression models from utils.py.

Operates on pandas DataFrames constructed from DB query results instead of CSVs.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# SEU category → normalization method
SEU_NORMS = {
    "Boiler": "hdd",
    "AHU": "hdd",
    "Catering": "opday",
    "Lighting": "opday",
    "AC / Refrigeration": "cdd",
    "Heater": "hdd",
    "ICT & Server Room Cooling": "fixed",
    "Electric Vehicles": "opday",
    "Solar": "pv",
    "Mains": "opday",
    "Other": "opday",
    "Unknown": "opday",
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


def get_seu_monthly(data: pd.DataFrame, summary_df: pd.DataFrame, climate_col: str, train_year: int, test_year: int):
    """Return monthly baseline/actual/predicted per SEU category for charting.

    Mirrors the SEU monthly plots from the original Streamlit app's tab4.
    """
    if data.empty:
        return []

    full_months = pd.Index(range(1, 13), name="month")
    months_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Order categories by actual consumption (comparison year), descending
    actual_by_seu = (
        data[data["year"] == test_year]
        .groupby("seu_category")["consumption"].sum()
        .sort_values(ascending=False)
    )
    categories = actual_by_seu.index.tolist()

    charts = []
    for category in categories:
        cat_data = data[data["seu_category"] == category].copy()
        if cat_data.empty:
            continue

        # Baseline monthly actuals
        baseline = cat_data[cat_data["year"] == train_year]
        baseline_monthly = baseline.groupby(baseline["date"].dt.month)["consumption"].sum().reindex(full_months, fill_value=0)

        # Comparison year monthly actuals
        actual = cat_data[cat_data["year"] == test_year]
        actual_monthly = actual.groupby(actual["date"].dt.month)["consumption"].sum().reindex(full_months, fill_value=0)

        # Predicted: run regression per meter, sum monthly predictions
        pred_monthly = np.zeros(12)
        for meter in cat_data["meter"].unique():
            meter_data = cat_data[cat_data["meter"] == meter].copy()
            train = meter_data[meter_data["year"] == train_year]
            test = meter_data[meter_data["year"] == test_year]
            if len(train) < 10 or len(test) < 1:
                continue
            # Align train to same date range as test
            max_test_date = test["date"].max()
            baseline_cutoff = pd.Timestamp(year=train_year, month=max_test_date.month, day=max_test_date.day)
            train = train[train["date"] <= baseline_cutoff]

            seu = meter_data["seu_category"].iloc[0] if pd.notna(meter_data["seu_category"].iloc[0]) else "Unknown"
            norm = SEU_NORMS.get(seu, "opday")

            if norm in ("hdd", "cdd") and climate_col in train.columns:
                train_clean = train.dropna(subset=[climate_col, "consumption"])
                test_clean = test.dropna(subset=[climate_col, "consumption"])
                if len(train_clean) < 10 or len(test_clean) < 1:
                    continue
                X_train = train_clean[[climate_col, "is_operational"]]
                model = LinearRegression().fit(X_train, train_clean["consumption"])
                test_clean = test_clean.copy()
                test_clean["predicted"] = model.predict(test_clean[[climate_col, "is_operational"]])
            elif norm not in ("fixed", "pv"):
                train_clean = train.dropna(subset=["consumption"])
                test_clean = test.dropna(subset=["consumption"])
                if len(train_clean) < 10 or len(test_clean) < 1:
                    continue
                X_train = train_clean[["is_operational"]]
                model = LinearRegression().fit(X_train, train_clean["consumption"])
                test_clean = test_clean.copy()
                test_clean["predicted"] = model.predict(test_clean[["is_operational"]])
            else:
                continue

            monthly_pred = test_clean.groupby(test_clean["date"].dt.month)["predicted"].sum().reindex(full_months, fill_value=0)
            pred_monthly += monthly_pred.values

        charts.append({
            "seu_category": category,
            "months": months_labels,
            "baseline": baseline_monthly.values.tolist(),
            "actual": actual_monthly.values.tolist(),
            "predicted": [float(v) for v in pred_monthly],
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


# ---------------------------------------------------------------------------
# ISO 50001 Energy Review engine (clauses 6.3 / 6.4 / 6.5)
# ---------------------------------------------------------------------------
TARIFFS_EUR_PER_KWH = {"electricity": 0.25, "gas": 0.09}
DEFAULT_TARIFF = 0.25

# Typical improvement measure + achievable saving fraction per SEU category
SEU_MEASURES = {
    "Boiler": ("Boiler optimisation, weather compensation & insulation", 0.12),
    "AHU": ("AHU scheduling, VSD fans & heat recovery", 0.15),
    "Heater": ("Heater controls, time scheduling & setpoint reduction", 0.10),
    "Catering": ("Catering equipment switch-off & efficient appliances", 0.08),
    "Lighting": ("LED retrofit with occupancy/daylight controls", 0.40),
    "AC / Refrigeration": ("Chiller/AC setpoint optimisation & maintenance", 0.12),
    "ICT & Server Room Cooling": ("Server room free-cooling & raised setpoints", 0.15),
    "Electric Vehicles": ("Off-peak smart charging", 0.05),
    "Mains": ("Base-load reduction & out-of-hours shutdown", 0.08),
    "Other": ("Out-of-hours shutdown & metering review", 0.05),
    "Unknown": ("Sub-meter and characterise consumption", 0.05),
}


def _r2(df: pd.DataFrame, cols: list[str]):
    d = df.dropna(subset=cols + ["consumption"])
    cols = [c for c in cols if d[c].nunique() > 1]
    if not cols or len(d) < 10 or d["consumption"].var() == 0:
        return None
    X, y = d[cols].astype(float), d["consumption"]
    return round(float(LinearRegression().fit(X, y).score(X, y)), 3)


def _tariff(utility: str) -> float:
    return TARIFFS_EUR_PER_KWH.get((utility or "").lower(), DEFAULT_TARIFF)


def build_energy_review(readings: list[dict], weather: list[dict], baseline_year: int, reporting_year: int,
                        pareto_threshold: float = 0.8) -> dict:
    """readings: dicts with date, meter, consumption, utility_type, seu_category."""
    empty = {"baseline_year": baseline_year, "reporting_year": reporting_year, "total_kwh": 0,
             "energy_sources": [], "seus": [], "opportunities": [], "enpi": [], "variables": []}
    if not readings:
        return empty
    df = pd.DataFrame(readings)
    df["date"] = pd.to_datetime(df["date"])
    df["consumption"] = pd.to_numeric(df["consumption"], errors="coerce").fillna(0)
    df["utility_type"] = df["utility_type"].fillna("unknown").str.lower()
    df["seu_category"] = df["seu_category"].fillna("Unknown")
    df["year"] = df["date"].dt.year
    if weather:
        w = pd.DataFrame(weather)
        w["date"] = pd.to_datetime(w["date"])
    else:
        w = pd.DataFrame(columns=["date", "hdd", "cdd"])

    rep = df[df["year"] == reporting_year]
    if rep.empty:
        return empty
    total = float(rep["consumption"].sum())

    # 1. Consumption by energy source
    sources = []
    for util, g in rep.groupby("utility_type"):
        kwh = float(g["consumption"].sum())
        sources.append({"source": util, "kwh": round(kwh, 1), "share": round(kwh / total, 4) if total else 0,
                        "cost_eur": round(kwh * _tariff(util), 2)})
    sources.sort(key=lambda s: -s["kwh"])

    # 2. Consumption by SEU category + Pareto
    def daily(sub):
        d = sub.groupby("date")["consumption"].sum().reset_index()
        d = d.merge(w[["date", "hdd", "cdd"]], on="date", how="left")
        d["production_day"] = (d["consumption"] > 0).astype(int)
        return d

    cats = rep.groupby("seu_category")["consumption"].sum().sort_values(ascending=False)
    seus, cum = [], 0.0
    for cat, kwh in cats.items():
        kwh = float(kwh)
        prev_cum = cum
        cum += kwh / total if total else 0
        sub = rep[rep["seu_category"] == cat]
        util_mix = sub.groupby("utility_type")["consumption"].sum()
        cost = float(sum(v * _tariff(u) for u, v in util_mix.items()))
        d_rep = daily(sub)
        r2 = {"hdd": _r2(d_rep, ["hdd"]), "cdd": _r2(d_rep, ["cdd"]),
              "production_days": _r2(d_rep, ["production_day"]),
              "combined": _r2(d_rep, ["hdd", "cdd", "production_day"])}
        relevant = [k for k in ("hdd", "cdd", "production_days") if r2[k] is not None and r2[k] >= 0.3]
        seus.append({
            "category": cat, "kwh": round(kwh, 1), "share": round(kwh / total, 4) if total else 0,
            "cumulative_share": round(cum, 4), "significant": prev_cum < pareto_threshold,
            "meters": int(sub["meter"].nunique()), "cost_eur": round(cost, 2),
            "blended_tariff": round(cost / kwh, 4) if kwh else DEFAULT_TARIFF,
            "utilities": sorted(util_mix.index.tolist()), "r2": r2, "relevant_variables": relevant,
        })

    # 3. EnPI: baseline vs reporting (normalised with baseline regression where possible)
    base = df[df["year"] == baseline_year]
    last = rep["date"].max()
    base = base[base["date"] <= pd.Timestamp(year=baseline_year, month=last.month, day=min(last.day, 28) if last.month == 2 else last.day)]
    enpi = []
    for s in seus:
        cat = s["category"]
        b_sub, r_sub = base[base["seu_category"] == cat], rep[rep["seu_category"] == cat]
        db, dr = daily(b_sub), daily(r_sub)
        b_kwh, r_kwh = float(db["consumption"].sum()), float(dr["consumption"].sum())
        b_days, r_days = int(db["production_day"].sum()), int(dr["production_day"].sum())
        expected = None
        cols = [c for c in ("hdd", "cdd") if c in s["relevant_variables"]] + ["production_day"]
        dbc = db.dropna(subset=cols)
        cols = [c for c in cols if dbc[c].nunique() > 1]
        drc = dr.dropna(subset=cols)
        if cols and len(dbc) >= 10 and len(drc) >= 1:
            m = LinearRegression().fit(dbc[cols].astype(float), dbc["consumption"])
            expected = float(np.clip(m.predict(drc[cols].astype(float)), 0, None).sum())
            expected += float(dr.loc[~dr.index.isin(drc.index), "consumption"].sum())
        b_int = b_kwh / b_days if b_days else None
        r_int = r_kwh / r_days if r_days else None
        enpi.append({
            "category": cat, "significant": s["significant"],
            "baseline_kwh": round(b_kwh, 1), "reporting_kwh": round(r_kwh, 1),
            "baseline_kwh_per_day": round(b_int, 1) if b_int else None,
            "reporting_kwh_per_day": round(r_int, 1) if r_int else None,
            "expected_kwh": round(expected, 1) if expected is not None else None,
            "enpi_ratio": round(r_kwh / expected, 3) if expected else None,
            "improvement_pct": round(100 * (1 - r_kwh / expected), 1) if expected else (
                round(100 * (1 - r_int / b_int), 1) if b_int and r_int else None),
            "model_variables": cols if expected is not None else [],
        })
    tb = float(sum(e["baseline_kwh"] for e in enpi))
    te = [e for e in enpi if e["expected_kwh"] is not None]
    t_exp = sum(e["expected_kwh"] for e in te) + sum(e["reporting_kwh"] for e in enpi if e["expected_kwh"] is None)
    totals = {"baseline_kwh": round(tb, 1), "reporting_kwh": round(total, 1), "expected_kwh": round(t_exp, 1),
              "enpi_ratio": round(total / t_exp, 3) if t_exp else None,
              "improvement_pct": round(100 * (1 - total / t_exp), 1) if t_exp else None,
              "change_pct": round(100 * (total - tb) / tb, 1) if tb else None}

    # 4. Improvement opportunities
    opps = []
    enpi_by = {e["category"]: e for e in enpi}
    for s in seus:
        measure, frac = SEU_MEASURES.get(s["category"], SEU_MEASURES["Other"])
        if s["category"] == "Solar":
            continue
        kwh = s["kwh"] * frac
        opps.append({"category": s["category"], "type": "measure", "title": measure,
                     "significant": s["significant"], "saving_pct": round(frac * 100, 1),
                     "kwh_savings": round(kwh, 0), "eur_savings": round(kwh * s["blended_tariff"], 0)})
        e = enpi_by[s["category"]]
        if e["expected_kwh"] and e["reporting_kwh"] > e["expected_kwh"] * 1.02:
            drift = e["reporting_kwh"] - e["expected_kwh"]
            opps.append({"category": s["category"], "type": "drift",
                         "title": f"Recover performance drift vs {baseline_year} baseline",
                         "significant": s["significant"],
                         "saving_pct": round(100 * drift / e["reporting_kwh"], 1),
                         "kwh_savings": round(drift, 0), "eur_savings": round(drift * s["blended_tariff"], 0)})
    opps.sort(key=lambda o: -o["eur_savings"])
    for i, o in enumerate(opps, 1):
        o["rank"] = i

    return {
        "baseline_year": baseline_year, "reporting_year": reporting_year,
        "total_kwh": round(total, 1),
        "total_cost_eur": round(sum(s["cost_eur"] for s in sources), 2),
        "pareto_threshold": pareto_threshold,
        "tariffs": TARIFFS_EUR_PER_KWH,
        "energy_sources": sources, "seus": seus, "enpi": enpi, "enpi_total": totals,
        "opportunities": opps,
        "total_kwh_savings": round(sum(o["kwh_savings"] for o in opps), 0),
        "total_eur_savings": round(sum(o["eur_savings"] for o in opps), 0),
    }
