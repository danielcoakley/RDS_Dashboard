from backend.storage_keys import report_storage_key, safe_key_part, upload_storage_key


def test_upload_storage_key_is_tenant_and_site_scoped():
    key = upload_storage_key(
        organization_id="Org 1",
        site_id="Main Site",
        upload_id="Upload 100",
        filename="Energy Data.csv",
    )

    assert key == "tenants/org-1/sites/main-site/uploads/upload-100/energy-data.csv"


def test_report_storage_key_is_run_scoped():
    key = report_storage_key(
        organization_id="org_1",
        site_id="site_1",
        run_id="run_1",
        filename="ISO Summary.json",
    )

    assert key == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"


def test_safe_key_part_rejects_blank_values():
    try:
        safe_key_part(" .. ")
        assert False, "Expected blank storage key part to fail"
    except ValueError as exc:
        assert "empty" in str(exc)
