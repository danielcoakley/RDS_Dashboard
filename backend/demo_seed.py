"""Synthetic demo tenant: every call creates a fresh, isolated organization.

Fully offline: weather (temperature, HDD/CDD) is generated, not fetched.
"""
import math
import random
import secrets
import uuid
from datetime import date, timedelta

from sqlalchemy.orm import Session

from auth import hash_password
from models import Organization, User, Site, Meter, EnergyReading, WeatherData, Objective
from routers.compliance import seed_compliance

HDD_BASE = 15.5
CDD_BASE = 18.0
DEMO_DAYS = 730

DEMO_METERS = [
    # name, utility_type, seu_category
    ("Compressor House", "electricity", "Compressed Air"),
    ("HVAC Plant", "electricity", "HVAC"),
    ("Lighting Distribution Board", "electricity", "Lighting"),
    ("Boiler House Gas", "gas", "Process Heating"),
]

DEMO_COMPLIANCE = {
    "4.1": ("complete", "Context analysis documented in EnMS manual section 2."),
    "4.2": ("complete", "Interested parties register maintained."),
    "4.3": ("complete", "EnMS scope: Cork manufacturing site, all energy types."),
    "4.4": ("in_progress", None),
    "5.1": ("complete", "Top management commitment statement signed."),
    "5.2": ("complete", "Energy policy approved and published on intranet."),
    "5.3": ("in_progress", None),
    "6.1": ("in_progress", None),
    "6.2": ("in_progress", None),
    "6.3": ("complete", "Energy review completed using platform SEU analysis."),
    "6.4": ("in_progress", None),
    "6.5": ("complete", "Weather-normalised baseline established from year 1 data."),
    "6.6": ("in_progress", None),
    "7.1": ("complete", "Energy team budget approved."),
    "7.2": ("in_progress", None),
    "7.5": ("in_progress", None),
    "8.1": ("in_progress", None),
}


def _synthetic_temp(d: date, rng: random.Random) -> tuple[float, float, float]:
    """Cork-like maritime climate: mean ~10.5C, warmest late July."""
    doy = d.timetuple().tm_yday
    seasonal = 10.5 + 5.5 * math.cos(2 * math.pi * (doy - 205) / 365.25)
    mean = seasonal + rng.gauss(0, 2.0)
    spread = 3.0 + rng.random() * 2.5
    return round(mean, 1), round(mean + spread, 1), round(mean - spread, 1)


def _consumption(seu: str, d: date, hdd: float, cdd: float, efficiency: float, rng: random.Random) -> float:
    weekday = d.weekday() < 5
    doy = d.timetuple().tm_yday
    if seu == "Compressed Air":
        base = 1800 if weekday else 650
        value = base + 6 * cdd
    elif seu == "HVAC":
        value = (320 if weekday else 140) + 22 * hdd + 60 * cdd
    elif seu == "Lighting":
        daylight = math.cos(2 * math.pi * (doy - 172) / 365.25)  # 1 at midsummer
        value = (430 if weekday else 120) * (1 - 0.25 * daylight)
    else:  # Process Heating (gas)
        value = (2600 if weekday else 700) + 190 * hdd
    noise = rng.gauss(1.0, 0.04)
    return round(max(value * efficiency * noise, 0.0), 1)


def create_demo_org(db: Session) -> User:
    rng = random.Random()
    org = Organization(name="Demo Manufacturing Ltd", sector="Manufacturing", country="Ireland")
    db.add(org)
    db.flush()

    user = User(
        org_id=org.id,
        name="Demo Energy Manager",
        email=f"demo-{uuid.uuid4().hex}@demo.enms.app",
        password_hash=hash_password(secrets.token_urlsafe(24)),
        role="admin",
    )
    db.add(user)

    site = Site(
        org_id=org.id, name="Cork Manufacturing Plant",
        address="Little Island, Cork, Ireland", latitude=51.9036, longitude=-8.3550,
        timezone="Europe/Dublin",
    )
    db.add(site)
    db.flush()

    meters = []
    for name, utility, seu in DEMO_METERS:
        m = Meter(site_id=site.id, name=name, utility_type=utility, units="kWh", seu_category=seu)
        db.add(m)
        meters.append(m)
    db.flush()

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=DEMO_DAYS - 1)
    weather_rows, reading_rows = [], []
    for i in range(DEMO_DAYS):
        d = start + timedelta(days=i)
        tmean, tmax, tmin = _synthetic_temp(d, rng)
        hdd = round(max(0.0, HDD_BASE - tmean), 2)
        cdd = round(max(0.0, tmean - CDD_BASE), 2)
        weather_rows.append(dict(site_id=site.id, date=d, temp_mean=tmean, temp_max=tmax,
                                 temp_min=tmin, hdd=hdd, cdd=cdd))
        # Second year shows ~6% improvement so baseline vs. reporting period has savings.
        efficiency = 1.0 if i < 365 else 0.94
        for m in meters:
            reading_rows.append(dict(
                meter_id=m.id, date=d, is_operational=d.weekday() < 5,
                consumption=_consumption(m.seu_category, d, hdd, cdd, efficiency, rng),
            ))
    db.bulk_insert_mappings(WeatherData, weather_rows)
    db.bulk_insert_mappings(EnergyReading, reading_rows)

    today = date.today()
    db.add_all([
        Objective(org_id=org.id, site_id=site.id, title="Reduce compressed air energy by 10%",
                  description="Fix leaks and lower header pressure by 0.5 bar.", target_pct=-10,
                  baseline_value=520000, current_value=489000,
                  deadline=today + timedelta(days=180), status="active"),
        Objective(org_id=org.id, site_id=site.id, title="Cut process heating gas intensity by 8%",
                  description="Boiler economiser and condensate return upgrade.", target_pct=-8,
                  baseline_value=1650000, current_value=1560000,
                  deadline=today + timedelta(days=300), status="active"),
        Objective(org_id=org.id, site_id=site.id, title="LED retrofit of production hall",
                  description="Replace high-bay metal halide fittings.", target_pct=-30,
                  baseline_value=120000, current_value=112000,
                  deadline=today + timedelta(days=60), status="at_risk"),
    ])

    seed_compliance(db, org.id, DEMO_COMPLIANCE)
    db.commit()
    db.refresh(user)
    return user
