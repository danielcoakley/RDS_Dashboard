from pydantic import BaseModel, EmailStr
from datetime import date
from typing import Optional, List


# --- Auth ---
class OrganizationCreate(BaseModel):
    name: str
    sector: Optional[str] = None
    country: Optional[str] = None

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    organization: OrganizationCreate

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    org_id: int
    class Config:
        from_attributes = True


# --- Sites ---
class SiteCreate(BaseModel):
    name: str
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = "UTC"

class SiteOut(BaseModel):
    id: int
    name: str
    address: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: str
    created_at: str
    meter_count: int = 0
    class Config:
        from_attributes = True

class SiteDetail(SiteOut):
    weather_status: Optional[str] = None
    latest_reading: Optional[str] = None


# --- Meters ---
class MeterCreate(BaseModel):
    name: str
    utility_type: str  # gas, electricity
    units: str = "kWh"
    seu_category: str = "Unknown"

class MeterOut(BaseModel):
    id: int
    name: str
    utility_type: str
    units: str
    seu_category: str
    created_at: str
    reading_count: int = 0
    class Config:
        from_attributes = True


# --- Analytics ---
class AnalysisRequest(BaseModel):
    site_id: int
    baseline_year: int
    comparison_year: int

class MeterSummary(BaseModel):
    meter: str
    seu_category: str
    baseline: float
    predicted: float
    actual: float
    estimated_savings: float
    pct_savings: Optional[float] = None
    baseline_days: int
    actual_days: int

class AnalysisSummary(BaseModel):
    gas: List[MeterSummary]
    electricity: List[MeterSummary]
    totals: dict

class SEUFlowNode(BaseModel):
    labels: List[str]
    sources: List[int]
    targets: List[int]
    values: List[float]


# --- Objectives ---
class ObjectiveCreate(BaseModel):
    title: str
    description: Optional[str] = None
    target_pct: Optional[float] = None
    baseline_value: Optional[float] = None
    deadline: Optional[date] = None
    site_id: Optional[int] = None

class ObjectiveOut(BaseModel):
    id: int
    title: str
    description: Optional[str]
    target_pct: Optional[float]
    baseline_value: Optional[float]
    current_value: Optional[float]
    deadline: Optional[date]
    status: str
    site_id: Optional[int]
    created_at: str
    class Config:
        from_attributes = True

class ObjectiveUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_pct: Optional[float] = None
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    deadline: Optional[date] = None
    status: Optional[str] = None


# --- Compliance ---
class ComplianceItemOut(BaseModel):
    id: Optional[int] = None
    clause_ref: str
    clause_title: str
    status: str
    evidence: Optional[str] = None
    class Config:
        from_attributes = True

class ComplianceUpdate(BaseModel):
    status: str
    evidence: Optional[str] = None


# --- Energy Review ---
class EnergyReviewCreate(BaseModel):
    site_id: int
    review_data: dict
    period_start: Optional[date] = None
    period_end: Optional[date] = None

class EnergyReviewOut(BaseModel):
    id: int
    site_id: int
    review_data: dict
    period_start: Optional[date]
    period_end: Optional[date]
    created_at: str
    class Config:
        from_attributes = True


# --- Weather ---
class WeatherStatus(BaseModel):
    site_id: int
    latest_date: Optional[str] = None
    total_days: int = 0
    status: str = "not_fetched"


# Fix forward references
TokenResponse.model_rebuild()
