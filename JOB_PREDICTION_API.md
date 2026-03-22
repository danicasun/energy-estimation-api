## Job Prediction FastAPI Service

This service provides energy and emissions estimates for a Slurm job using the local power model and ElectricityMaps carbon intensity data.

### Requirements
- Python 3.10+
- Dependencies from `requirements.txt` (includes `fastapi` and `uvicorn`)

### Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export ELECTRICITYMAPS_API_KEY="your_key"
uvicorn job_prediction_api:app --host 0.0.0.0 --port 8001
```

### Request schema (units)
- `sbatchText` (optional): raw SBATCH content.
- `parameters` (optional):
  - `cpuCores` (count)
  - `gpuCount` (count)
  - `memoryGigabytes` (GB)
  - `walltimeHours` (hours)
  - `partitionName` (string)
  - `nodeCount` (count)
- `zone` (optional): ElectricityMaps zone override.

Example:
```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "cpuCores": 8,
      "gpuCount": 1,
      "memoryGigabytes": 32,
      "walltimeHours": 2
    }
  }'
```

### Response schema (units)
- `energy_kwh` (kWh)
- `emissions_kgco2e` (kg CO2e)
- `carbon_intensity_gco2e_per_kwh` (g CO2e/kWh)
- `power_watts` (W)

### Next.js proxy integration
The Job Forecast tab calls `/api/job-prediction`, which proxies to the Python API. Configure the proxy with:
```bash
export PYTHON_JOB_PREDICTION_URL="http://127.0.0.1:8001/predict"
```
If unset, the proxy defaults to `http://127.0.0.1:8001/predict`.

### systemd setup
1. Copy the unit file:
   ```bash
   sudo cp deploy/systemd/job-prediction-api.service /etc/systemd/system/
   ```
2. Edit the unit file to set:
   - `WorkingDirectory`
   - `PYTHONPATH`
   - `ELECTRICITYMAPS_API_KEY`
   - `ExecStart` (venv path)
   - `User` and `Group` (recommended)
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable job-prediction-api
   sudo systemctl start job-prediction-api
   sudo systemctl status job-prediction-api
   ```
