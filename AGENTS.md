# Notes for agents

Streamlit dashboard (`app.py` + `utils.py`). No database, no external credentials.

- Run: `docker compose -f docker-compose.base44.yml up -d` (Streamlit on host port 3000).
- Deps install at container start (`requirements.txt`); pip cache is a named volume, so restarts are fast.
- `pyinstaller` in requirements is only for the Windows .exe build (`build.py`), not needed at runtime.
- The dashboard shows "Please upload all required files" until CSVs are uploaded in the sidebar; `seu_mapping.csv` (repo root and `Data/`) is one of the sample inputs.
- Streamlit rejects websockets when Origin != Host, which breaks the proxied preview. An nginx service (`.base44/nginx.conf`) fronts port 3000 and rewrites Host/Origin to `localhost:8501`; Streamlit itself is not published directly.
