from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from database import engine, Base
from routers import auth, sites, data, analytics, objectives, compliance, energy_review, team, audit, reports

# Create tables on startup
Base.metadata.create_all(bind=engine)

# Lightweight idempotent migrations for existing databases
with engine.begin() as conn:
    if engine.dialect.name == "postgresql":  # fresh SQLite DBs get the column from create_all
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS team_role VARCHAR(50)"))
    conn.execute(text("UPDATE users SET role = 'owner' WHERE role = 'admin'"))
    conn.execute(text("UPDATE users SET role = 'manager' WHERE role = 'member'"))

app = FastAPI(title="EnMS — ISO 50001 Energy Management Platform")

# CORS — allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(sites.router)
app.include_router(data.router)
app.include_router(analytics.router)
app.include_router(objectives.router)
app.include_router(compliance.router)
app.include_router(energy_review.router)
app.include_router(reports.router)
app.include_router(team.router)
app.include_router(audit.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
