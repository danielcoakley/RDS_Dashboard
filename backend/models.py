import os
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date, Text,
    ForeignKey, UniqueConstraint, JSON
)
from sqlalchemy.orm import relationship
from database import Base


class Organization(Base):
    __tablename__ = "organizations"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    sector = Column(String(255))
    country = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    users = relationship("User", back_populates="organization")
    sites = relationship("Site", back_populates="organization", cascade="all, delete-orphan")
    objectives = relationship("Objective", back_populates="organization", cascade="all, delete-orphan")
    compliance_items = relationship("ComplianceItem", back_populates="organization", cascade="all, delete-orphan")
    energy_reviews = relationship("EnergyReview", back_populates="organization", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="admin")  # admin, member
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="users")


class Site(Base):
    __tablename__ = "sites"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    timezone = Column(String(100), default="UTC")
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="sites")
    meters = relationship("Meter", back_populates="site", cascade="all, delete-orphan")
    weather_data = relationship("WeatherData", back_populates="site", cascade="all, delete-orphan")
    objectives = relationship("Objective", back_populates="site")
    energy_reviews = relationship("EnergyReview", back_populates="site")
    uploads = relationship("DataUpload", back_populates="site", cascade="all, delete-orphan")


class Meter(Base):
    __tablename__ = "meters"
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    name = Column(String(255), nullable=False)
    utility_type = Column(String(50), nullable=False)  # gas, electricity
    units = Column(String(50), default="kWh")
    seu_category = Column(String(255), default="Unknown")
    created_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("Site", back_populates="meters")
    readings = relationship("EnergyReading", back_populates="meter", cascade="all, delete-orphan")


class EnergyReading(Base):
    __tablename__ = "energy_readings"
    id = Column(Integer, primary_key=True, index=True)
    meter_id = Column(Integer, ForeignKey("meters.id"), nullable=False)
    date = Column(Date, nullable=False)
    consumption = Column(Float, nullable=False)
    is_operational = Column(Boolean, default=True)

    meter = relationship("Meter", back_populates="readings")
    __table_args__ = (UniqueConstraint("meter_id", "date", name="uq_meter_date"),)


class WeatherData(Base):
    __tablename__ = "weather_data"
    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    date = Column(Date, nullable=False)
    temp_mean = Column(Float)
    temp_max = Column(Float)
    temp_min = Column(Float)
    hdd = Column(Float)
    cdd = Column(Float)

    site = relationship("Site", back_populates="weather_data")
    __table_args__ = (UniqueConstraint("site_id", "date", name="uq_site_date_weather"),)


class DataUpload(Base):
    """Tracks files uploaded for a site (energy data + SEU mappings)."""
    __tablename__ = "data_uploads"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    filename = Column(String(500), nullable=False)
    kind = Column(String(50), nullable=False)  # energy, seu_mapping
    records = Column(Integer, default=0)
    detail = Column(String(255))
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("Site", back_populates="uploads")


class EnergyReview(Base):
    __tablename__ = "energy_reviews"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=False)
    review_data = Column(JSON, default=dict)  # structured energy review content
    period_start = Column(Date)
    period_end = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="energy_reviews")
    site = relationship("Site", back_populates="energy_reviews")


class Objective(Base):
    __tablename__ = "objectives"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    target_pct = Column(Float)  # e.g. -10 for 10% reduction
    baseline_value = Column(Float)
    current_value = Column(Float)
    deadline = Column(Date)
    status = Column(String(50), default="active")  # active, completed, at_risk
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="objectives")
    site = relationship("Site", back_populates="objectives")


class ComplianceItem(Base):
    __tablename__ = "compliance_items"
    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    clause_ref = Column(String(20), nullable=False)  # e.g. "6.3"
    clause_title = Column(String(255))
    status = Column(String(50), default="not_started")  # not_started, in_progress, complete
    evidence = Column(Text)
    responsible_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    organization = relationship("Organization", back_populates="compliance_items")
