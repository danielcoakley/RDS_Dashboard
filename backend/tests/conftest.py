import os
import sys
import tempfile

import pytest

_DB_FILE = os.path.join(tempfile.mkdtemp(), "test_enms.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_FILE}"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


def signup(client, email, org_name="Org"):
    r = client.post("/api/auth/signup", json={
        "name": "Test User", "email": email, "password": "secret-pass-123",
        "organization": {"name": org_name, "sector": "Manufacturing", "country": "Ireland"},
    })
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
