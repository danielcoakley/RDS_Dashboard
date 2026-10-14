"""Builds a fresh, isolated demo organization with synthetic (offline) data."""
import math
import random
import uuid
from datetime import date, timedelta
from sqlalchemy.orm import Session
from models import Organization, User, Site, Meter, EnergyReading, WeatherData, Objective
from auth import hash_password
from iso_clauses import seed_compliance

DAYS = 730
HDD_BASE, CDD_BASE = 15.5, 18.0

# name, utility, seu, base kWh/day, kWh per HDD, kWh per CDD
METERS = [
    ("Compressed Air", "electricity", "Compressed Air", 1800, 0, 25),
    ("HVAC", "electricity", "HVAC", 900, 35, 160),
    ("Lighting", "electricity", "Lighting", 600, 6, 0),
    ("Process Heating", "gas", "Process Heating", 3200, 260, 0),
]

DEMO_STATUSES = {
    "4.1": "complete", "4.2": "complete", "4.3": "complete", "4.4": "in_progress",
    "5.1": "complete", "5.2": "complete", "5.3": "in_progress",
    "6.1": "in_progress", "6.2": "complete", "6.3": "complete", "6.4": "in_progress",
    "6.5": "complete", "6.6": "in_progress",
    "7.1": "complete", "7.2": "in_progress", "7.5": "in_progress",
    "8.1": "in_progress",
    "9.1": "in_progress",
}
DEMO_EVIDENCE = {
    "4.1": "Context register EnMS-REG-01 approved by Ops Director.",
    "5.2": "Energy policy signed by MD, posted on site noticeboards.",
    "6.3": "Energy review completed: 4 SEUs identified (Process Heating, Compressed Air, HVAC, Lighting).",
    "6.5": "Weather-normalised regression baseline set on year one data.",
    "7.1": "Energy manager appointed; metering budget approved.",
}


def _weather(start: date, rng: random.Random):
    rows = []
    for i in range(DAYS):
        d = start + timedelta(days=i)
        doy = d.timetuple().tm_yday
        # Cork: ~6°C mid-winter, ~16°C mid-summer, with noise and occasional warm spells
        mean = 11 - 5.5 * math.cos(2 * math.pi * (doy - 15) / 365) + rng.gauss(0, 2.2)
        if 160 < doy < 240 and rng.random() < 0.12:
            mean += rng.uniform(3, 6)
        mean = round(mean, 1)
        rows.append({
            "date": d, "temp_mean": mean,
            "temp_max": round(mean + rng.uniform(2, 5), 1),
            "temp_min": round(mean - rng.uniform(2, 5), 1),
            "hdd": round(max(0.0, HDD_BASE - mean), 2),
            "cdd": round(max(0.0, mean - CDD_BASE), 2),
        })
    return rows


def create_demo_org(db: Session) -> User:
    rng = random.Random()
    start = date(date.today().year - 2, 1, 1)
    year_two = start.year + 1

    org = Organization(name="Demo Manufacturing Ltd", sector="Manufacturing", country="Ireland")
    db.add(org)
    db.flush()
    user = User(
        org_id=org.id, name="Demo User", email=f"demo-{uuid.uuid4()}@demo.enms.app",
        password_hash=hash_password(uuid.uuid4().hex), role="admin",
    )
    site = Site(org_id=org.id, name="Cork Plant", address="Cork, Ireland",
                latitude=51.8985, longitude=-8.4756, timezone="Europe/Dublin")
    db.add_all([user, site])
    db.flush()

    weather = _weather(start, rng)
    db.bulk_insert_mappings(WeatherData, [{**w, "site_id": site.id} for w in weather])

    for name, utility, seu, base, k_hdd, k_cdd in METERS:
        m = Meter(site_id=site.id, name=name, utility_type=utility, units="kWh", seu_category=seu)
        db.add(m)
        db.flush()
        readings = []
        for w in weather:
            d = w["date"]
            operational = d.weekday() < 5
            load = base * (1.0 if operational else 0.35)
            value = load + k_hdd * w["hdd"] + k_cdd * w["cdd"]
            if d.year == year_two:
                value *= 0.94
            value *= rng.gauss(1, 0.04)
            readings.append({"meter_id": m.id, "date": d, "consumption": round(max(value, 0), 1),
                             "is_operational": operational})
        db.bulk_insert_mappings(EnergyReading, readings)

    db.add_all([
        Objective(org_id=org.id, site_id=site.id, title="Reduce Process Heating gas by 8%",
                  description="Boiler economiser and steam trap programme.", target_pct=-8,
                  deadline=date(date.today().year, 12, 31), status="active"),
        Objective(org_id=org.id, site_id=site.id, title="Cut compressed air leaks by 30%",
                  description="Quarterly ultrasonic leak surveys.", target_pct=-5,
                  deadline=date(date.today().year, 9, 30), status="at_risk"),
        Objective(org_id=org.id, site_id=site.id, title="LED retrofit of production hall",
                  description="Replace high-bay metal halide fittings.", target_pct=-40,
                  deadline=date(date.today().year - 1, 12, 31), status="completed"),
    ])
    seed_compliance(db, org.id, DEMO_STATUSES, DEMO_EVIDENCE)
    db.commit()
    db.refresh(user)
    return user
