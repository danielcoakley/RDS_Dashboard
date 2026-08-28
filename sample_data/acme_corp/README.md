# Acme Corp Demo Data

This folder contains anonymized showcase inputs for the Streamlit dashboard.

The files preserve the source structure, dates, consumption values, weather values, and SEU categories needed to run the analysis. Client, site, area, meter, and weather-station labels have been replaced with generic Acme Corp names.

Run the demo locally with:

```powershell
python -m streamlit run app.py --server.headless true --server.port 8501 --client.toolbarMode minimal
```

Then open:

```text
http://localhost:8501/?demo=acme
```
