import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.artifacts import InMemoryTenantArtifactStore
from backend.csp_reports import (
    CSP_REPORT_MAX_BYTES,
    CSP_REPORT_RETENTION_DAYS_ENV,
    CspViolationReportPayloadError,
    CspViolationReportTooLargeError,
    apply_csp_violation_report_retention,
    build_csp_violation_report,
    csp_report_retention_days_from_env,
    process_csp_violation_report,
)
from backend.database import initialize_database
from backend.domain import (
    CspBlockedUriCategory,
    CspViolationReportDirective,
    CspViolationReportDisposition,
)
from backend.store import SaaSStore
from backend.support import SUPPORT_ACCESS_TOKEN_ENV


def legacy_csp_report_payload(**overrides) -> bytes:
    report = {
        "document-uri": "https://tenant.example.com/dashboard?token=secret",
        "referrer": "https://tenant.example.com/login?email=owner@example.com",
        "blocked-uri": "https://cdn.bad.example/assets/app.js?org=org_1",
        "violated-directive": "script-src 'self'",
        "effective-directive": "script-src-elem",
        "original-policy": "script-src 'self'; report-uri /security/csp-reports",
        "source-file": "https://tenant.example.com/_next/static/app.js?build=secret",
        "script-sample": "alert('owner@example.com')",
        "disposition": "report",
        "status-code": 200,
    }
    report.update(overrides)
    return json.dumps({"csp-report": report}, separators=(",", ":")).encode("utf-8")


def test_build_csp_violation_report_sanitizes_browser_payload():
    report = build_csp_violation_report(
        legacy_csp_report_payload(),
        now=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )

    persisted_values = [
        report.id,
        report.disposition.value,
        report.effective_directive.value,
        report.violated_directive.value,
        report.blocked_uri_category.value,
        report.document_origin,
        str(report.status_code),
    ]
    assert report.disposition == CspViolationReportDisposition.REPORT
    assert report.effective_directive == CspViolationReportDirective.SCRIPT_SRC
    assert report.violated_directive == CspViolationReportDirective.SCRIPT_SRC
    assert report.blocked_uri_category == CspBlockedUriCategory.HTTPS
    assert report.document_origin == "https://tenant.example.com"
    assert report.status_code == 200
    assert report.created_at.isoformat() == "2026-07-20T09:00:00+00:00"
    assert "owner@example.com" not in " ".join(persisted_values)
    assert "/dashboard" not in " ".join(persisted_values)
    assert "cdn.bad.example" not in " ".join(persisted_values)
    assert "org_1" not in " ".join(persisted_values)


def test_build_csp_violation_report_accepts_reporting_api_body_shape():
    payload = json.dumps(
        {
            "type": "csp-violation",
            "url": "https://app.example.com/reports/report_1?download=true",
            "body": {
                "effective_directive": "connect-src",
                "blocked_uri": "http://legacy.example.com/export.csv",
                "disposition": "enforce",
                "status_code": "0",
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")

    report = build_csp_violation_report(payload)

    assert report.disposition == CspViolationReportDisposition.ENFORCE
    assert report.effective_directive == CspViolationReportDirective.CONNECT_SRC
    assert report.violated_directive == CspViolationReportDirective.CONNECT_SRC
    assert report.blocked_uri_category == CspBlockedUriCategory.HTTP
    assert report.document_origin == "https://app.example.com"
    assert report.status_code == 0


def test_process_csp_violation_report_stores_only_sanitized_fields():
    store = SaaSStore(initialize_database())

    report = process_csp_violation_report(store, legacy_csp_report_payload())

    row = store.conn.execute(
        "SELECT * FROM csp_violation_reports WHERE id = ?",
        (report.id,),
    ).fetchone()
    stored_text = " ".join(str(value) for value in dict(row).values())
    assert row["disposition"] == "report"
    assert row["effective_directive"] == "script-src"
    assert row["blocked_uri_category"] == "https"
    assert row["document_origin"] == "https://tenant.example.com"
    assert "owner@example.com" not in stored_text
    assert "/dashboard" not in stored_text
    assert "cdn.bad.example" not in stored_text
    assert "org_1" not in stored_text
    assert store.count_csp_violation_reports_by_disposition() == {
        "enforce": 0,
        "report": 1,
    }
    assert store.count_csp_violation_reports_by_effective_directive()["script-src"] == 1


def test_process_csp_violation_report_rejects_invalid_payloads():
    store = SaaSStore(initialize_database())

    for payload in (b"", b"not-json", b"[]"):
        try:
            process_csp_violation_report(store, payload)
            assert False, "Expected invalid CSP report payload to fail"
        except CspViolationReportPayloadError:
            pass

    try:
        process_csp_violation_report(store, b"{" + (b'"x":' + b'"a"' * CSP_REPORT_MAX_BYTES))
        assert False, "Expected oversized CSP report payload to fail"
    except CspViolationReportTooLargeError:
        pass


def test_csp_report_retention_days_from_env_validates_policy(monkeypatch):
    monkeypatch.delenv(CSP_REPORT_RETENTION_DAYS_ENV, raising=False)

    assert csp_report_retention_days_from_env() == 30

    monkeypatch.setenv(CSP_REPORT_RETENTION_DAYS_ENV, "45")

    assert csp_report_retention_days_from_env() == 45

    for value in ("0", "3651", "not-a-number"):
        monkeypatch.setenv(CSP_REPORT_RETENTION_DAYS_ENV, value)
        try:
            csp_report_retention_days_from_env()
            assert False, "Expected invalid CSP retention days to fail"
        except ValueError as exc:
            assert CSP_REPORT_RETENTION_DAYS_ENV in str(exc)


def test_apply_csp_report_retention_deletes_only_reports_older_than_cutoff():
    store = SaaSStore(initialize_database())
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(),
        now=datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
    )
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(**{"effective-directive": "connect-src"}),
        now=datetime(2026, 6, 20, 9, 0, tzinfo=timezone.utc),
    )
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(**{"effective-directive": "img-src"}),
        now=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
    )

    summary = apply_csp_violation_report_retention(
        store,
        retention_days=30,
        now=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )
    remaining = store.list_csp_violation_reports(limit=10)

    assert summary.retention_days == 30
    assert summary.cutoff_created_before == "2026-06-20T09:00:00+00:00"
    assert summary.deleted_count == 1
    assert summary.remaining_count == 2
    assert [row["effective_directive"] for row in remaining] == ["img-src", "connect-src"]


def test_csp_violation_report_route_accepts_public_browser_report():
    store = SaaSStore(initialize_database())
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/security/csp-reports",
        content=legacy_csp_report_payload(),
        headers={"Content-Type": "application/csp-report"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["report_id"].startswith("csp_report_")
    assert payload["effective_directive"] == "script-src"
    assert payload["blocked_uri_category"] == "https"
    assert "organization_id" not in payload
    assert "document_origin" not in payload
    assert "tenant.example.com" not in response.text
    assert "owner@example.com" not in response.text
    assert store.count_csp_violation_reports_by_disposition()["report"] == 1


def test_csp_violation_report_route_rejects_malformed_and_large_reports():
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    malformed = client.post(
        "/security/csp-reports",
        content=b"not-json",
        headers={"Content-Type": "application/csp-report"},
    )
    too_large = client.post(
        "/security/csp-reports",
        content=b'{"blocked-uri":"' + (b"a" * CSP_REPORT_MAX_BYTES) + b'"}',
        headers={"Content-Type": "application/csp-report"},
    )

    assert malformed.status_code == 400
    assert too_large.status_code == 413


def test_support_csp_reports_route_requires_support_token(monkeypatch):
    monkeypatch.delenv(SUPPORT_ACCESS_TOKEN_ENV, raising=False)
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    missing_config = client.get("/support/csp-reports")
    missing_summary_config = client.get("/support/csp-reports/summary")
    missing_retention_config = client.post("/support/csp-reports/retention/apply")

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    missing_auth = client.get("/support/csp-reports")
    missing_summary_auth = client.get("/support/csp-reports/summary")
    missing_retention_auth = client.post("/support/csp-reports/retention/apply")
    bad_auth = client.get(
        "/support/csp-reports",
        headers={"Authorization": "Bearer wrong-secret"},
    )
    bad_summary_auth = client.get(
        "/support/csp-reports/summary",
        headers={"Authorization": "Bearer wrong-secret"},
    )
    bad_retention_auth = client.post(
        "/support/csp-reports/retention/apply",
        headers={"Authorization": "Bearer wrong-secret"},
    )

    assert missing_config.status_code == 503
    assert missing_summary_config.status_code == 503
    assert missing_retention_config.status_code == 503
    assert SUPPORT_ACCESS_TOKEN_ENV in missing_config.json()["detail"]
    assert missing_auth.status_code == 401
    assert missing_summary_auth.status_code == 401
    assert missing_retention_auth.status_code == 401
    assert missing_auth.headers["www-authenticate"] == "Bearer"
    assert bad_auth.status_code == 401
    assert bad_summary_auth.status_code == 401
    assert bad_retention_auth.status_code == 401


def test_support_csp_reports_route_lists_filtered_sanitized_reports(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(),
        now=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(
            **{
                "document-uri": "https://app.example.com/reports/report_1?org=org_1",
                "effective-directive": "connect-src",
                "violated-directive": "connect-src",
                "blocked-uri": "http://legacy.example.com/export.csv?tenant=org_1",
                "disposition": "enforce",
                "status-code": 0,
            }
        ),
        now=datetime(2026, 7, 20, 9, 1, tzinfo=timezone.utc),
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer support-secret"}

    all_reports = client.get("/support/csp-reports", headers=headers)
    filtered = client.get(
        "/support/csp-reports?disposition=report&directive=script-src",
        headers=headers,
    )
    unsupported = client.get(
        "/support/csp-reports?directive=script-src-elem",
        headers=headers,
    )

    assert all_reports.status_code == 200
    reports = all_reports.json()
    assert [report["effective_directive"] for report in reports] == [
        "connect-src",
        "script-src",
    ]
    assert reports[0]["document_origin"] == "https://app.example.com"
    assert reports[0]["blocked_uri_category"] == "http"
    assert reports[0]["status_code"] == 0
    assert filtered.status_code == 200
    assert len(filtered.json()) == 1
    assert filtered.json()[0]["effective_directive"] == "script-src"
    assert filtered.json()[0]["document_origin"] == "https://tenant.example.com"
    assert unsupported.status_code == 400
    assert "Unsupported CSP report directive" in unsupported.json()["detail"]

    serialized = all_reports.text + filtered.text
    assert "/dashboard" not in serialized
    assert "/reports/report_1" not in serialized
    assert "cdn.bad.example" not in serialized
    assert "legacy.example.com" not in serialized
    assert "owner@example.com" not in serialized
    assert "org_1" not in serialized


def test_support_csp_report_retention_route_applies_configured_policy(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    monkeypatch.setenv(CSP_REPORT_RETENTION_DAYS_ENV, "1")
    store = SaaSStore(initialize_database())
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(),
        now=datetime(2000, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(
            **{
                "document-uri": "https://app.example.com/settings?org=org_1",
                "effective-directive": "img-src",
                "blocked-uri": "https://cdn.example.com/logo.png?tenant=org_1",
            }
        ),
        now=datetime(2999, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/support/csp-reports/retention/apply",
        headers={"Authorization": "Bearer support-secret"},
    )
    remaining = client.get(
        "/support/csp-reports",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["retention_days"] == 1
    assert payload["deleted_count"] == 1
    assert payload["remaining_count"] == 1
    assert remaining.status_code == 200
    assert [report["effective_directive"] for report in remaining.json()] == ["img-src"]
    serialized = response.text + remaining.text
    assert "/settings" not in serialized
    assert "cdn.example.com" not in serialized
    assert "org_1" not in serialized


def test_support_csp_reports_summary_returns_finite_safe_counts(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(),
        now=datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc),
    )
    process_csp_violation_report(
        store,
        legacy_csp_report_payload(
            **{
                "document-uri": "https://app.example.com/settings?org=org_1",
                "effective-directive": "connect-src",
                "violated-directive": "connect-src",
                "blocked-uri": "http://legacy.example.com/export.csv?tenant=org_1",
                "disposition": "enforce",
                "status-code": 0,
            }
        ),
        now=datetime(2026, 7, 20, 9, 1, tzinfo=timezone.utc),
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get(
        "/support/csp-reports/summary",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    payload = response.json()
    dispositions = {
        item["disposition"]: item["count"]
        for item in payload["by_disposition"]
    }
    directives = {
        item["directive"]: item["count"]
        for item in payload["by_effective_directive"]
    }
    assert payload["total_count"] == 2
    assert payload["latest_report_created_at"] == "2026-07-20T09:01:00+00:00"
    assert dispositions == {"enforce": 1, "report": 1}
    assert directives["script-src"] == 1
    assert directives["connect-src"] == 1
    assert directives["default-src"] == 0
    assert len(directives) == len(CspViolationReportDirective)

    serialized = response.text
    assert "tenant.example.com" not in serialized
    assert "app.example.com" not in serialized
    assert "/settings" not in serialized
    assert "legacy.example.com" not in serialized
    assert "owner@example.com" not in serialized
    assert "org_1" not in serialized
