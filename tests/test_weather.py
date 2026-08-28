from datetime import date
import json
from io import BytesIO

import pytest

from backend import weather


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info) -> None:
        return None


def test_geocode_location_returns_coordinates(monkeypatch):
    def fake_urlopen(request, timeout=10.0):
        assert request.headers["User-agent"] == weather.USER_AGENT
        return _FakeResponse([{"lat": "51.5074", "lon": "-0.1278"}])

    monkeypatch.setattr(weather, "urlopen", fake_urlopen)

    latitude, longitude = weather.geocode_location("London, UK")

    assert latitude == pytest.approx(51.5074)
    assert longitude == pytest.approx(-0.1278)


def test_geocode_location_raises_when_no_results(monkeypatch):
    monkeypatch.setattr(weather, "urlopen", lambda request, timeout=10.0: _FakeResponse([]))

    with pytest.raises(ValueError, match="No location found"):
        weather.geocode_location("Nowhere Land")


def test_geocode_location_requires_non_empty_query():
    with pytest.raises(ValueError):
        weather.geocode_location("   ")


def test_fetch_daily_degree_days_computes_hdd_and_cdd(monkeypatch):
    payload = {
        "daily": {
            "time": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "temperature_2m_max": [10.0, 20.0, 30.0],
            "temperature_2m_min": [0.0, 10.0, 20.0],
        }
    }
    monkeypatch.setattr(weather, "urlopen", lambda request, timeout=15.0: _FakeResponse(payload))

    frame = weather.fetch_daily_degree_days(
        51.5,
        -0.1,
        date(2025, 1, 1),
        date(2025, 1, 3),
        hdd_base_temperature_c=15.5,
        cdd_base_temperature_c=22.0,
    )

    # mean temps: 5, 15, 25
    assert list(frame["mean_temperature_c"]) == pytest.approx([5.0, 15.0, 25.0])
    assert list(frame["hdd"]) == pytest.approx([10.5, 0.5, 0.0])
    assert list(frame["cdd"]) == pytest.approx([0.0, 0.0, 3.0])


def test_fetch_daily_degree_days_rejects_inverted_range():
    with pytest.raises(ValueError, match="start_date must not be after end_date"):
        weather.fetch_daily_degree_days(51.5, -0.1, date(2025, 1, 3), date(2025, 1, 1))


def test_fetch_daily_degree_days_raises_when_no_data_returned(monkeypatch):
    monkeypatch.setattr(
        weather, "urlopen", lambda request, timeout=15.0: _FakeResponse({"daily": {"time": []}})
    )

    with pytest.raises(ValueError, match="No historical weather data"):
        weather.fetch_daily_degree_days(51.5, -0.1, date(2025, 1, 1), date(2025, 1, 3))


def test_summarise_degree_days_totals(monkeypatch):
    payload = {
        "daily": {
            "time": ["2025-01-01", "2025-01-02"],
            "temperature_2m_max": [10.0, 30.0],
            "temperature_2m_min": [0.0, 20.0],
        }
    }
    monkeypatch.setattr(weather, "urlopen", lambda request, timeout=15.0: _FakeResponse(payload))
    frame = weather.fetch_daily_degree_days(51.5, -0.1, date(2025, 1, 1), date(2025, 1, 2))

    summary = weather.summarise_degree_days(
        frame, hdd_base_temperature_c=15.5, cdd_base_temperature_c=22.0
    )

    assert summary.days == 2
    assert summary.start_date == "2025-01-01"
    assert summary.end_date == "2025-01-02"
    assert summary.total_hdd == pytest.approx(10.5)
    assert summary.total_cdd == pytest.approx(3.0)
