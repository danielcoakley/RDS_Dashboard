# EnMS — ISO 50001 Energy Management Platform

Full rebuild from the original Streamlit dashboard into a modern Next.js + FastAPI app.

## Architecture
- **Frontend**: Next.js 14 (App Router) + TypeScript + Tailwind CSS + Plotly charts — in `frontend/`
- **Backend**: FastAPI (Python) with scikit-learn/pandas analytics — in `backend/`
- **Database**: PostgreSQL 16
- **Proxy**: Nginx fronts port 3000, routing `/api/*` to the backend and everything else to the frontend

## Running
```bash
docker compose -f docker-compose.base44.yml up -d
```
- Frontend: http://localhost:3000 (landing page at `/`, app at `/dashboard`)
- Backend API: http://localhost:3000/api/ (proxied) or http://localhost:8000/api/ (direct)

## Key design decisions
- **Single-origin**: nginx proxies API calls through port 3000 — no CORS issues, no separate API URL config
- **Weather data**: Open-Meteo Archive API (free, no key) — HDD (base 15.5°C) and CDD (base 18°C) calculated from daily mean temps
- **Geocoding**: Open-Meteo Geocoding API — site address → lat/long for weather fetch
- **Auth**: JWT-based, email/password, organization-scoped (72h token expiry)
- **Analytics**: Regression models ported from original `utils.py` — per-meter OLS with HDD/CDD/operational-day features

## Backend structure
- `main.py` — FastAPI app, CORS, router includes, auto-creates tables
- `models.py` — SQLAlchemy models (Organization, User, Site, Meter, EnergyReading, WeatherData, EnergyReview, Objective, ComplianceItem)
- `schemas.py` — Pydantic schemas
- `auth.py` — JWT + bcrypt password hashing
- `analytics.py` — Regression models ported from original Streamlit utils.py
- `weather.py` — Open-Meteo integration (weather fetch + geocoding)
- `routers/` — auth, sites, data (upload + weather), analytics, objectives, compliance, energy_review

## Frontend structure
- `src/app/page.tsx` — Landing page (marketing)
- `src/app/(auth)/` — login, signup
- `src/app/(app)/` — dashboard, sites, baseline, seu-analysis, energy-review, objectives, compliance
- `src/lib/api.ts` — API client (uses relative URLs via nginx proxy)
- `src/lib/auth.tsx` — Auth context with JWT

## CSV upload formats
1. Simple: `date,meter,consumption`
2. Original RDS: `Metered Sector, Utility, Units, Period, [date columns DD/MM/YYYY]`

## Verifying the app
- `curl http://localhost:3000/api/health` → `{"status":"ok"}`
- Create account via `/signup`, then navigate to `/dashboard`
- Create a site, upload CSV data, fetch weather, run baseline analysis

## Known issues / notes
- `bcrypt` must be pinned to 4.0.1 (passlib + bcrypt 4.1+ compatibility bug)
- Plotly charts loaded via `next/dynamic` (ssr:false) to avoid SSR issues
- The original Streamlit files (`app.py`, `utils.py`, etc.) are kept for reference but not used
