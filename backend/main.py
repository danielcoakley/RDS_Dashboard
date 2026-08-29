from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import auth, sites, data, analytics, objectives, compliance, energy_review

# Create tables on startup
Base.metadata.create_all(bind=engine)

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


@app.get("/api/health")
def health():
    return {"status": "ok"}
