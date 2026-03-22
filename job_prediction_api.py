"""FastAPI service for job energy/emissions predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from electricitymaps import get_carbon_intensity
from power_model import estimate_emissions
from sbatch_parser import SbatchParameters, parse_sbatch_text
from slurm_runtime import get_node_prefix
from zone_mapping import get_zone_for_node_prefix


class JobPredictionError(Exception):
    """Raised when prediction inputs are invalid."""


class JobPredictionParameters(BaseModel):
    partitionName: Optional[str] = None
    nodeCount: Optional[int] = None
    cpuCores: Optional[int] = None
    gpuCount: Optional[int] = None
    memoryGigabytes: Optional[float] = None
    walltimeHours: Optional[float] = None


class JobPredictionRequest(BaseModel):
    sbatchText: Optional[str] = None
    parameters: Optional[JobPredictionParameters] = None
    zone: Optional[str] = None


class JobPredictionResponse(BaseModel):
    energy_kwh: float
    emissions_kgco2e: float
    emissions_gco2e: float
    power_watts: float
    carbon_intensity_gco2e_per_kwh: float
    zone: str
    inputs: Dict[str, Any]
    notes: List[str]


def _parse_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sbatch_text = payload.get("sbatchText") or ""
    parameters = payload.get("parameters") or {}

    parsed_sbatch: Optional[SbatchParameters] = None
    if isinstance(sbatch_text, str) and sbatch_text.strip():
        parsed_sbatch = parse_sbatch_text(sbatch_text)

    cpu_cores = _coalesce_int(
        parameters.get("cpuCores"),
        _calculate_total_cpu_cores(parsed_sbatch),
        1,
    )
    gpu_count = _coalesce_int(parameters.get("gpuCount"), parsed_sbatch.gpu_count if parsed_sbatch else None, 0)
    node_count = _coalesce_int(parameters.get("nodeCount"), parsed_sbatch.node_count if parsed_sbatch else None, None)
    partition_name = _coalesce_str(
        parameters.get("partitionName"),
        parsed_sbatch.partition_name if parsed_sbatch else None,
    )

    memory_gigabytes = _coalesce_float(
        parameters.get("memoryGigabytes"),
        _calculate_total_memory_gigabytes(parsed_sbatch, cpu_cores),
        0.0,
    )
    walltime_hours = _coalesce_float(
        parameters.get("walltimeHours"),
        parsed_sbatch.walltime_hours if parsed_sbatch else None,
        1.0,
    )

    zone_override = payload.get("zone")
    if zone_override is not None and not isinstance(zone_override, str):
        raise JobPredictionError("zone must be a string when provided.")

    zone = zone_override or _resolve_zone(parsed_sbatch)

    return {
        "cpu_cores": cpu_cores,
        "gpu_count": gpu_count,
        "node_count": node_count,
        "partition_name": partition_name,
        "memory_gigabytes": memory_gigabytes,
        "walltime_hours": walltime_hours,
        "zone": zone,
    }


def _calculate_total_cpu_cores(parsed_sbatch: Optional[SbatchParameters]) -> Optional[int]:
    if not parsed_sbatch:
        return None
    cpus_per_task = parsed_sbatch.cpu_cores_per_task
    task_count = parsed_sbatch.task_count
    if cpus_per_task is None:
        return None
    return cpus_per_task * (task_count or 1)


def _calculate_total_memory_gigabytes(
    parsed_sbatch: Optional[SbatchParameters],
    cpu_cores: int,
) -> Optional[float]:
    if not parsed_sbatch:
        return None
    if parsed_sbatch.memory_gigabytes_total is not None:
        return parsed_sbatch.memory_gigabytes_total
    if parsed_sbatch.memory_gigabytes_per_cpu is not None:
        return parsed_sbatch.memory_gigabytes_per_cpu * max(cpu_cores, 1)
    return None


def _resolve_zone(parsed_sbatch: Optional[SbatchParameters]) -> str:
    node_list = parsed_sbatch.nodelist if parsed_sbatch else None
    node_prefix = get_node_prefix(node_list) if node_list else None
    return get_zone_for_node_prefix(node_prefix)


def _coalesce_int(*values: Optional[Any]) -> Optional[int]:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            raise JobPredictionError("Invalid integer value in request.")
    return None


def _coalesce_float(*values: Optional[Any]) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            raise JobPredictionError("Invalid numeric value in request.")
    return None


def _coalesce_str(*values: Optional[Any]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str):
            raise JobPredictionError("Invalid string value in request.")
        return value
    return None


def _build_prediction_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
    carbon_intensity = get_carbon_intensity(inputs["zone"])
    if carbon_intensity is None:
        carbon_intensity = 0.0

    results = estimate_emissions(
        inputs["cpu_cores"],
        inputs["memory_gigabytes"],
        inputs["walltime_hours"],
        carbon_intensity,
    )

    notes = []
    if inputs.get("gpu_count", 0) > 0:
        notes.append("GPU power is not included in the current power model.")

    return {
        "energy_kwh": results["energy_kwh"],
        "emissions_kgco2e": results["emissions_kgco2e"],
        "emissions_gco2e": results["emissions_gco2e"],
        "power_watts": results["power_watts"],
        "carbon_intensity_gco2e_per_kwh": carbon_intensity,
        "zone": inputs["zone"],
        "calculation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cpu_cores": inputs["cpu_cores"],
            "gpu_count": inputs["gpu_count"],
            "node_count": inputs["node_count"],
            "partition_name": inputs["partition_name"],
            "memory_gigabytes": inputs["memory_gigabytes"],
            "walltime_hours": inputs["walltime_hours"],
        },
        "notes": notes,
    }


app = FastAPI(title="Job Prediction API")


@app.post("/predict", response_model=JobPredictionResponse)
def predict_job(request: JobPredictionRequest) -> JobPredictionResponse:
    payload = _model_to_dict(request)
    try:
        inputs = _parse_request_payload(payload)
        response_body = _build_prediction_response(inputs)
    except JobPredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safeguard
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return JobPredictionResponse(**response_body)


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("job_prediction_api:app", host="0.0.0.0", port=8001)
