"""Smart upload helpers: CSV/XLSX parsing, column detection, data-quality report, sample data."""
import io
import numpy as np
import pandas as pd
from datetime import date, timedelta

TEMPLATE_CSV = "date,meter,consumption\n2025-01-01,Main Electricity,1250.5\n2025-01-01,Boiler Gas,3400\n2025-01-02,Main Electricity,1198.2\n2025-01-02,Boiler Gas,3550\n"


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    df.columns = [str(c).strip().strip("'").strip('"') for c in df.columns]
    return df


def is_rds_format(df: pd.DataFrame) -> bool:
    return "Metered Sector" in df.columns and "Utility" in df.columns


def detect_columns(df: pd.DataFrame) -> dict:
    """Pick the most date-like, most numeric, and a text 'meter' column."""
    sample = df.head(200)
    best_date, best_date_score = None, 0.0
    best_val, best_val_score = None, 0.0
    meter_col = None
    for c in df.columns:
        s = sample[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            num_score, date_score = 0.0, 1.0
        else:
            num_score = pd.to_numeric(s, errors="coerce").notna().mean()
            date_score = 0.0
        if num_score < 0.5 and date_score == 0.0:
            date_score = pd.to_datetime(s.astype(str), errors="coerce", dayfirst=True).notna().mean()
        lname = c.lower()
        if any(k in lname for k in ("date", "day", "time", "period")):
            date_score += 0.1
        if any(k in lname for k in ("kwh", "consumption", "value", "usage", "reading")):
            num_score += 0.1
        if date_score > best_date_score:
            best_date, best_date_score = c, date_score
        if num_score > best_val_score and num_score >= 0.6:
            best_val, best_val_score = c, num_score
        if meter_col is None and num_score < 0.5 and date_score < 0.5 and any(
            k in lname for k in ("meter", "name", "sector", "utility", "channel")
        ):
            meter_col = c
    if best_date_score < 0.6:
        best_date = None
    return {"date_col": best_date, "value_col": best_val, "meter_col": meter_col}


def normalize(df: pd.DataFrame, date_col: str, value_col: str, meter_col: str | None, meter_name: str) -> pd.DataFrame:
    for c in (date_col, value_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")
    out = pd.DataFrame({
        "date": (pd.to_datetime(df[date_col], errors="coerce") if pd.api.types.is_datetime64_any_dtype(df[date_col])
                 else pd.to_datetime(df[date_col].astype(str), errors="coerce", dayfirst=True)).dt.date,
        "value": pd.to_numeric(df[value_col], errors="coerce"),
        "meter": df[meter_col].astype(str).str.strip() if meter_col and meter_col in df.columns else meter_name,
    })
    return out.dropna(subset=["date", "value"])


def quality_report(norm: pd.DataFrame, total_rows: int) -> dict:
    meters = []
    for name, g in norm.groupby("meter"):
        dates = pd.to_datetime(g["date"])
        dmin, dmax = dates.min(), dates.max()
        expected = pd.date_range(dmin, dmax, freq="D")
        missing = expected.difference(pd.DatetimeIndex(dates.unique()))
        dup_mask = g.duplicated(subset=["date"], keep=False)
        neg = g[g["value"] < 0]
        vals = g["value"]
        std = vals.std(ddof=0)
        z = (vals - vals.mean()) / std if std and std > 0 else vals * 0
        outl = g[z.abs() > 4]
        meters.append({
            "meter": name,
            "rows": int(len(g)),
            "start": dmin.date().isoformat(),
            "end": dmax.date().isoformat(),
            "missing_days": int(len(missing)),
            "missing_sample": [d.date().isoformat() for d in missing[:10]],
            "duplicates": int(dup_mask.sum()),
            "duplicate_sample": sorted({d.isoformat() for d in g[dup_mask]["date"]})[:10],
            "negatives": int(len(neg)),
            "negative_sample": [{"date": r.date.isoformat(), "value": float(r.value)} for r in neg.head(10).itertuples()],
            "outliers": int(len(outl)),
            "outlier_sample": [
                {"date": r.date.isoformat(), "value": float(r.value), "z": round(float(z.loc[r.Index]), 2)}
                for r in outl.head(10).itertuples()
            ],
        })
    issues = sum(m["missing_days"] + m["duplicates"] + m["negatives"] + m["outliers"] for m in meters)
    return {
        "total_rows": total_rows,
        "valid_rows": int(len(norm)),
        "unparseable_rows": int(total_rows - len(norm)),
        "issue_count": int(issues),
        "meters": meters,
    }


def synthetic_series(utility_type: str, days: int = 730, seed: int = 0) -> list[tuple[date, float]]:
    """Realistic daily kWh: seasonal heating for gas, weekday pattern for electricity."""
    rng = np.random.default_rng(seed)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    out = []
    for i in range(days):
        d = start + timedelta(days=i)
        doy = d.timetuple().tm_yday
        winter = (np.cos(2 * np.pi * (doy - 15) / 365.25) + 1) / 2  # 1 mid-Jan, 0 mid-Jul
        weekend = d.weekday() >= 5
        trend = 1 - 0.03 * (i / 365)  # ~3%/yr efficiency gain
        if utility_type == "gas":
            v = (400 + 2600 * winter ** 1.5) * (0.75 if weekend else 1.0)
        else:
            v = (1500 + 300 * winter + 200 * (1 - winter) ** 3) * (0.55 if weekend else 1.0)
        v *= trend * rng.normal(1.0, 0.06)
        out.append((d, round(float(max(v, 0.0)), 1)))
    return out
