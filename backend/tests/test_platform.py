from conftest import signup
from routers.compliance import compute_readiness, ISO_50001_CLAUSES


def test_demo_returns_token_and_isolated_orgs(client):
    r1 = client.post("/api/auth/demo")
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["user"]["email"].endswith("@demo.enms.app")
    h1 = {"Authorization": f"Bearer {body['access_token']}"}

    sites = client.get("/api/sites", headers=h1).json()
    assert len(sites) == 1
    meters = client.get(f"/api/sites/{sites[0]['id']}/meters", headers=h1).json()
    assert len(meters) == 4
    summary = client.get(f"/api/data/summary/{sites[0]['id']}", headers=h1)
    assert summary.status_code == 200, summary.text

    readiness = client.get("/api/compliance/readiness", headers=h1).json()
    assert readiness["total"] == 26 and 0 < readiness["score"] < 100

    r2 = client.post("/api/auth/demo").json()
    assert r2["user"]["org_id"] != body["user"]["org_id"]
    h2 = {"Authorization": f"Bearer {r2['access_token']}"}
    assert client.get(f"/api/sites/{sites[0]['id']}", headers=h2).status_code == 404


def test_readiness_math():
    statuses = {"4.1": "complete", "4.2": "complete", "4.3": "in_progress", "5.1": "complete"}
    r = compute_readiness(statuses)
    assert r["total"] == 26
    assert r["complete"] == 3 and r["in_progress"] == 1 and r["not_started"] == 22
    assert r["score"] == round(100 * 3.5 / 26)
    sec4 = r["by_section"][0]
    assert sec4 == {"section": "4", "title": "Context of the organization", "score": round(100 * 2.5 / 4), "complete": 2, "total": 4}
    assert [s["section"] for s in r["by_section"]] == ["4", "5", "6", "7", "8", "9", "10"]
    assert [a["clause_ref"] for a in r["next_actions"]] == ["4.3", "4.4", "5.2", "5.3", "6.1"]
    assert r["next_actions"][0]["status"] == "in_progress"
    assert compute_readiness({c["clause_ref"]: "complete" for c in ISO_50001_CLAUSES})["score"] == 100


def test_signup_seeds_compliance_and_readiness_endpoint(client):
    h = signup(client, "seed@example.com")
    r = client.get("/api/compliance/readiness", headers=h).json()
    assert r == {**r, "score": 0, "complete": 0, "in_progress": 0, "not_started": 26, "total": 26}
    assert client.put("/api/compliance/6.3", json={"status": "complete"}, headers=h).status_code == 200
    assert client.get("/api/compliance/readiness", headers=h).json()["complete"] == 1
    assert client.get("/api/compliance/readiness").status_code == 401


def test_tenant_isolation_sites_and_compliance(client):
    ha = signup(client, "a@example.com", "Org A")
    hb = signup(client, "b@example.com", "Org B")
    site = client.post("/api/sites", json={"name": "A Plant", "latitude": 51.9, "longitude": -8.4}, headers=ha)
    assert site.status_code == 200, site.text
    sid = site.json()["id"]
    meter = client.post(f"/api/sites/{sid}/meters", json={"name": "M1", "utility_type": "gas"}, headers=ha)
    assert meter.status_code == 200, meter.text

    assert client.get("/api/sites", headers=hb).json() == []
    assert client.get(f"/api/sites/{sid}", headers=hb).status_code == 404
    assert client.get(f"/api/sites/{sid}/meters", headers=hb).status_code == 404
    assert client.delete(f"/api/sites/{sid}/meters/{meter.json()['id']}", headers=hb).status_code == 404
    assert client.post("/api/objectives", json={"title": "x", "site_id": sid}, headers=hb).status_code == 404

    client.put("/api/compliance/5.2", json={"status": "complete", "evidence": "A policy"}, headers=ha)
    b_items = {i["clause_ref"]: i for i in client.get("/api/compliance", headers=hb).json()}
    assert b_items["5.2"]["status"] == "not_started" and b_items["5.2"]["evidence"] is None
    assert client.get("/api/compliance/readiness", headers=hb).json()["complete"] == 0


def test_demo_data_feeds_analytics(client):
    h = {"Authorization": f"Bearer {client.post('/api/auth/demo').json()['access_token']}"}
    sid = client.get("/api/sites", headers=h).json()[0]["id"]
    r = client.get(f"/api/analytics/years/{sid}", headers=h)
    assert r.status_code == 200, r.text
    assert len(r.json()["years"]) >= 2
