from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import User, Site, Meter, EnergyReading, WeatherData
from schemas import AnalysisRequest, MeterSummary, AnalysisSummary, SEUFlowNode
from auth import get_current_user
from database import get_db
from analytics import build_dataframe, evaluate_meter_models, get_monthly_comparison, build_sankey, get_seu_monthly
import pandas as pd

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _compute_all(gas_df, elec_df, baseline_year, comparison_year):
    """Compute every analysis output from the loaded data — runs once per selection."""
    gas_summary = evaluate_meter_models(gas_df, baseline_year, comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, baseline_year, comparison_year) if not elec_df.empty else pd.DataFrame()

    total_gas_pred = float(gas_summary["predicted"].sum()) if not gas_summary.empty else 0
    total_gas_actual = float(gas_summary["actual"].sum()) if not gas_summary.empty else 0
    total_elec_pred = float(elec_summary["predicted"].sum()) if not elec_summary.empty else 0
    total_elec_actual = float(elec_summary["actual"].sum()) if not elec_summary.empty else 0

    analysis = {
        "gas": [MeterSummary(**r) for r in gas_summary.to_dict("records")] if not gas_summary.empty else [],
        "electricity": [MeterSummary(**r) for r in elec_summary.to_dict("records")] if not elec_summary.empty else [],
        "totals": {
            "gas_predicted": round(total_gas_pred, 1),
            "gas_actual": round(total_gas_actual, 1),
            "gas_savings_pct": round(100 * (total_gas_pred - total_gas_actual) / total_gas_pred, 1) if total_gas_pred else 0,
            "elec_predicted": round(total_elec_pred, 1),
            "elec_actual": round(total_elec_actual, 1),
            "elec_savings_pct": round(100 * (total_elec_pred - total_elec_actual) / total_elec_pred, 1) if total_elec_pred else 0,
        },
    }

    monthly = {
        "gas": get_monthly_comparison(gas_df, gas_summary, "hdd", baseline_year, comparison_year) if not gas_df.empty else [],
        "electricity": get_monthly_comparison(elec_df, elec_summary, "cdd", baseline_year, comparison_year) if not elec_df.empty else [],
    }

    total_gas = total_gas_actual
    total_elec = total_elec_actual
    sankey = build_sankey(gas_summary, elec_summary, total_gas, total_elec)

    def aggregate_by_seu(summary_df):
        if summary_df.empty:
            return []
        grouped = summary_df.groupby("seu_category").agg({
            "baseline": "sum", "predicted": "sum", "actual": "sum",
            "estimated_savings": "sum", "baseline_days": "sum", "actual_days": "sum",
        }).reset_index()
        grouped["pct_savings"] = (100 * grouped["estimated_savings"] / grouped["actual"]).round(1).where(grouped["actual"] != 0, None)
        return grouped.to_dict("records")

    seu_summary = {
        "gas": aggregate_by_seu(gas_summary),
        "electricity": aggregate_by_seu(elec_summary),
    }
    seu_monthly = {
        "gas": get_seu_monthly(gas_df, gas_summary, "hdd", baseline_year, comparison_year) if not gas_df.empty else [],
        "electricity": get_seu_monthly(elec_df, elec_summary, "cdd", baseline_year, comparison_year) if not elec_df.empty else [],
    }
    return {"analysis": analysis, "monthly": monthly, "sankey": sankey, "seuSummary": seu_summary, "seuMonthly": seu_monthly}


def _load_site_data(db: Session, site_id: int, org_id: int):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    meters = db.query(Meter).filter(Meter.site_id == site_id).all()
    weather = db.query(WeatherData).filter(WeatherData.site_id == site_id).all()
    weather_rows = [{"date": w.date, "hdd": w.hdd, "cdd": w.cdd} for w in weather]

    gas_dfs, elec_dfs = [], []
    for m in meters:
        readings = db.query(EnergyReading).filter(EnergyReading.meter_id == m.id).all()
        reading_rows = [{"date": r.date, "consumption": r.consumption} for r in readings]
        df = build_dataframe(reading_rows, weather_rows, m.seu_category, m.utility_type)
        if df.empty:
            continue
        df["meter"] = m.name
        if m.utility_type.lower() in ("gas", "gas"):
            gas_dfs.append(df)
        else:
            elec_dfs.append(df)

    gas_df = pd.concat(gas_dfs, ignore_index=True) if gas_dfs else pd.DataFrame()
    elec_df = pd.concat(elec_dfs, ignore_index=True) if elec_dfs else pd.DataFrame()
    return site, meters, gas_df, elec_df


@router.post("/analysis", response_model=AnalysisSummary)
def run_analysis(payload: AnalysisRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site, meters, gas_df, elec_df = _load_site_data(db, payload.site_id, user.org_id)

    gas_summary = evaluate_meter_models(gas_df, payload.baseline_year, payload.comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, payload.baseline_year, payload.comparison_year) if not elec_df.empty else pd.DataFrame()

    total_gas_pred = float(gas_summary["predicted"].sum()) if not gas_summary.empty else 0
    total_gas_actual = float(gas_summary["actual"].sum()) if not gas_summary.empty else 0
    total_elec_pred = float(elec_summary["predicted"].sum()) if not elec_summary.empty else 0
    total_elec_actual = float(elec_summary["actual"].sum()) if not elec_summary.empty else 0

    return AnalysisSummary(
        gas=[MeterSummary(**r) for r in gas_summary.to_dict("records")] if not gas_summary.empty else [],
        electricity=[MeterSummary(**r) for r in elec_summary.to_dict("records")] if not elec_summary.empty else [],
        totals={
            "gas_predicted": round(total_gas_pred, 1),
            "gas_actual": round(total_gas_actual, 1),
            "gas_savings_pct": round(100 * (total_gas_pred - total_gas_actual) / total_gas_pred, 1) if total_gas_pred else 0,
            "elec_predicted": round(total_elec_pred, 1),
            "elec_actual": round(total_elec_actual, 1),
            "elec_savings_pct": round(100 * (total_elec_pred - total_elec_actual) / total_elec_pred, 1) if total_elec_pred else 0,
        },
    )


@router.get("/years/{site_id}")
def available_years(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)
    years = set()
    if not gas_df.empty:
        years |= set(gas_df["year"].unique())
    if not elec_df.empty:
        years |= set(elec_df["year"].unique())
    return {"years": sorted(int(y) for y in years)}


@router.get("/monthly/{site_id}")
def monthly_comparison(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)

    gas_summary = evaluate_meter_models(gas_df, baseline_year, comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, baseline_year, comparison_year) if not elec_df.empty else pd.DataFrame()

    gas_charts = get_monthly_comparison(gas_df, gas_summary, "hdd", baseline_year, comparison_year) if not gas_df.empty else []
    elec_charts = get_monthly_comparison(elec_df, elec_summary, "cdd", baseline_year, comparison_year) if not elec_df.empty else []

    return {"gas": gas_charts, "electricity": elec_charts}


@router.get("/sankey/{site_id}")
def sankey_diagram(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)

    gas_summary = evaluate_meter_models(gas_df, baseline_year, comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, baseline_year, comparison_year) if not elec_df.empty else pd.DataFrame()

    total_gas = float(gas_summary["actual"].sum()) if not gas_summary.empty else 0
    total_elec = float(elec_summary["actual"].sum()) if not elec_summary.empty else 0

    flow = build_sankey(gas_summary, elec_summary, total_gas, total_elec)
    return flow


@router.get("/seu-summary/{site_id}")
def seu_summary(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)

    gas_summary = evaluate_meter_models(gas_df, baseline_year, comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, baseline_year, comparison_year) if not elec_df.empty else pd.DataFrame()

    def aggregate_by_seu(summary_df):
        if summary_df.empty:
            return []
        grouped = summary_df.groupby("seu_category").agg({
            "baseline": "sum", "predicted": "sum", "actual": "sum",
            "estimated_savings": "sum", "baseline_days": "sum", "actual_days": "sum",
        }).reset_index()
        grouped["pct_savings"] = (100 * grouped["estimated_savings"] / grouped["actual"]).round(1).where(grouped["actual"] != 0, None)
        return grouped.to_dict("records")

    return {
        "gas": aggregate_by_seu(gas_summary),
        "electricity": aggregate_by_seu(elec_summary),
    }


@router.get("/seu-monthly/{site_id}")
def seu_monthly(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Monthly baseline/actual/predicted per SEU category — mirrors the original app's SEU charts."""
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)

    gas_summary = evaluate_meter_models(gas_df, baseline_year, comparison_year) if not gas_df.empty else pd.DataFrame()
    elec_summary = evaluate_meter_models(elec_df, baseline_year, comparison_year) if not elec_df.empty else pd.DataFrame()

    gas_charts = get_seu_monthly(gas_df, gas_summary, "hdd", baseline_year, comparison_year) if not gas_df.empty else []
    elec_charts = get_seu_monthly(elec_df, elec_summary, "cdd", baseline_year, comparison_year) if not elec_df.empty else []

    return {"gas": gas_charts, "electricity": elec_charts}


@router.get("/bundle/{site_id}")
def analysis_bundle(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """All analysis outputs in one call — loads site data once instead of 5 separate requests."""
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)
    return _compute_all(gas_df, elec_df, baseline_year, comparison_year)
