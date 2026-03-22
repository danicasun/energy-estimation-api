# Job Prediction API

FastAPI service that estimates job energy (kWh) and emissions (kg CO2e) using a simple power model and ElectricityMaps carbon intensity (g CO2e/kWh).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ELECTRICITYMAPS_API_KEY="your_key"
python -m uvicorn job_prediction_api:app --host 0.0.0.0 --port 8001
```

Open `http://127.0.0.1:8001/docs`.

## Request/response units
- Energy: kWh
- Carbon intensity: g CO2e/kWh
- Emissions: kg CO2e
- Walltime: hours
- Calculation time: UTC (ISO 8601)

See `JOB_PREDICTION_API.md` for request examples and systemd setup.
