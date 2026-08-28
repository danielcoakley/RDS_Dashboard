import pandas as pd

from utils import preprocess_data


def _weather_df(label):
    return pd.DataFrame(
        {
            "Description:": [
                "Source:",
                "Accuracy:",
                "Station:",
                "Station ID:",
                None,
                "Date",
                "01/01/2024",
                "02/01/2024",
            ],
            label: [
                "example",
                "example",
                "example",
                "example",
                None,
                label,
                4.0,
                5.0,
            ],
        }
    )


def _energy_df(gas_meter="Boiler gas", elec_meter="Office power"):
    return pd.DataFrame(
        {
            "Metered Sector": [gas_meter, None, elec_meter, None],
            "Utility": ["Gas", None, "Electrical", None],
            "Units": ["kWh", None, "kWh", None],
            "Period": ["Total", "Day", "Total", "Day"],
            "01/01/2024": [11, 10, 21, 20],
            "02/01/2024": [13, 12, 23, 22],
        }
    )


def test_preprocess_uses_uploaded_seu_mapping_without_root_file():
    energy = _energy_df()
    mapping = pd.DataFrame(
        {
            "Meter": ["Boiler gas", "Office power"],
            "SEU_Category": ["Boiler", "Lighting"],
        }
    )

    gas_df, elec_df = preprocess_data(
        energy,
        _weather_df("HDD 15.5"),
        _weather_df("CDD 15.5"),
        seu_mapping_df=mapping,
        seu_mapping_path="does-not-exist.csv",
    )

    assert gas_df["SEU_Category"].unique().tolist() == ["Boiler"]
    assert elec_df["SEU_Category"].unique().tolist() == ["Lighting"]


def test_preprocess_filters_ignored_uploaded_meters():
    energy = _energy_df(elec_meter="Ignored power")
    mapping = pd.DataFrame(
        {
            "Meter": ["Boiler gas", "Ignored power"],
            "SEU_Category": ["Boiler", "Ignore"],
        }
    )

    gas_df, elec_df = preprocess_data(
        energy,
        _weather_df("HDD 15.5"),
        _weather_df("CDD 15.5"),
        seu_mapping_df=mapping,
        seu_mapping_path="does-not-exist.csv",
    )

    assert gas_df["Meter"].unique().tolist() == ["Boiler gas"]
    assert elec_df.empty


def test_preprocess_without_mapping_does_not_read_repo_default_mapping():
    energy = _energy_df(gas_meter="EM33 Members", elec_meter="EM24 Industries")

    gas_df, elec_df = preprocess_data(
        energy,
        _weather_df("HDD 15.5"),
        _weather_df("CDD 15.5"),
    )

    assert gas_df["SEU_Category"].unique().tolist() == ["Unknown"]
    assert elec_df["SEU_Category"].unique().tolist() == ["Unknown"]


def test_preprocess_can_use_explicit_mapping_path(tmp_path):
    energy = _energy_df()
    mapping_path = tmp_path / "tenant_seu_mapping.csv"
    mapping_path.write_text(
        "Meter,SEU_Category\nBoiler gas,Boiler\nOffice power,Lighting\n",
        encoding="utf-8",
    )

    gas_df, elec_df = preprocess_data(
        energy,
        _weather_df("HDD 15.5"),
        _weather_df("CDD 15.5"),
        seu_mapping_path=mapping_path,
    )

    assert gas_df["SEU_Category"].unique().tolist() == ["Boiler"]
    assert elec_df["SEU_Category"].unique().tolist() == ["Lighting"]


def test_preprocess_rejects_invalid_explicit_mapping():
    energy = _energy_df()
    mapping = pd.DataFrame({"Meter": ["Boiler gas"]})

    try:
        preprocess_data(
            energy,
            _weather_df("HDD 15.5"),
            _weather_df("CDD 15.5"),
            seu_mapping_df=mapping,
        )
        assert False, "Expected invalid tenant SEU mapping to fail"
    except ValueError as exc:
        assert "SEU_Category" in str(exc)
