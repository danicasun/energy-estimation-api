"""FastAPI service for job energy/emissions predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from energy_constants import POWER_USAGE_EFFECTIVENESS
from electricitymaps import get_carbon_intensity
from node_inventory import aggregate_allocated_nodes, lookup_node_profile
from power_model import estimate_emissions
from sbatch_parser import SbatchParameters, parse_sbatch_text
from slurm_runtime import expand_slurm_nodelist, get_node_prefix, get_slurm_env_vars
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
    nodelist: Optional[str] = None


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
    pue: float
    zone: str
    inputs: Dict[str, Any]
    notes: List[str]


def _parse_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    sbatch_text = payload.get("sbatchText") or ""
    parameters = payload.get("parameters") or {}

    parsed_sbatch: Optional[SbatchParameters] = None
    if isinstance(sbatch_text, str) and sbatch_text.strip():
        parsed_sbatch = parse_sbatch_text(sbatch_text)

    resolved_nodelist = _resolve_nodelist(parameters, parsed_sbatch)
    allocated_node_names = expand_slurm_nodelist(resolved_nodelist)
    node_inventory_aggregation = aggregate_allocated_nodes(allocated_node_names)

    inferred_cpu_cores = _calculate_total_cpu_cores(parsed_sbatch)
    if inferred_cpu_cores is None and node_inventory_aggregation.total_cpu_cores > 0:
        inferred_cpu_cores = node_inventory_aggregation.total_cpu_cores

    cpu_cores = _coalesce_int(parameters.get("cpuCores"), inferred_cpu_cores, 1)

    inferred_gpu_count = parsed_sbatch.gpu_count if parsed_sbatch else None
    if inferred_gpu_count is None and node_inventory_aggregation.total_gpu_count > 0:
        inferred_gpu_count = node_inventory_aggregation.total_gpu_count
    gpu_count = _coalesce_int(parameters.get("gpuCount"), inferred_gpu_count, 0)

    node_count = _coalesce_int(
        parameters.get("nodeCount"),
        parsed_sbatch.node_count if parsed_sbatch else None,
        len(allocated_node_names) if allocated_node_names else None,
    )
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

    zone = zone_override or _resolve_zone(resolved_nodelist, parsed_sbatch)

    return {
        "cpu_cores": cpu_cores,
        "gpu_count": gpu_count,
        "node_count": node_count,
        "partition_name": partition_name,
        "memory_gigabytes": memory_gigabytes,
        "walltime_hours": walltime_hours,
        "zone": zone,
        "resolved_nodelist": resolved_nodelist,
        "allocated_node_names": allocated_node_names,
        "allocated_node_cpu_cores": _build_allocated_node_cpu_core_list(allocated_node_names),
        "missing_inventory_node_names": node_inventory_aggregation.missing_node_names,
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


def _resolve_zone(
    resolved_nodelist: Optional[str],
    parsed_sbatch: Optional[SbatchParameters],
) -> str:
    node_list = resolved_nodelist or (parsed_sbatch.nodelist if parsed_sbatch else None)
    node_prefix = get_node_prefix(node_list) if node_list else None
    return get_zone_for_node_prefix(node_prefix)


def _resolve_nodelist(
    parameters: Dict[str, Any],
    parsed_sbatch: Optional[SbatchParameters],
) -> Optional[str]:
    parameter_nodelist = parameters.get("nodelist")
    if parameter_nodelist is not None:
        if not isinstance(parameter_nodelist, str):
            raise JobPredictionError("Invalid string value in request.")
        if parameter_nodelist.strip():
            return parameter_nodelist.strip()

    if parsed_sbatch and parsed_sbatch.nodelist:
        return parsed_sbatch.nodelist

    runtime_nodelist = get_slurm_env_vars().get("nodelist")
    if runtime_nodelist and runtime_nodelist.strip():
        return runtime_nodelist.strip()

    return None


def _build_allocated_node_cpu_core_list(allocated_node_names: List[str]) -> Optional[List[int]]:
    allocated_node_cpu_cores: List[int] = []
    for node_name in allocated_node_names:
        profile = lookup_node_profile(node_name)
        if profile is None:
            continue
        allocated_node_cpu_cores.append(max(profile.cpu_core_count, 0))
    return allocated_node_cpu_cores or None


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
        allocated_node_cpu_cores=inputs.get("allocated_node_cpu_cores"),
    )

    notes = []
    if inputs.get("gpu_count", 0) > 0:
        notes.append("GPU power is not included in the current power model.")
    if inputs.get("missing_inventory_node_names"):
        notes.append(
            "Node inventory lookup missing for: "
            + ", ".join(inputs["missing_inventory_node_names"])
            + ". Generic fallback assumptions were used."
        )

    return {
        "energy_kwh": results["energy_kwh"],
        "emissions_kgco2e": results["emissions_kgco2e"],
        "emissions_gco2e": results["emissions_gco2e"],
        "power_watts": results["power_watts"],
        "carbon_intensity_gco2e_per_kwh": carbon_intensity,
        "pue": POWER_USAGE_EFFECTIVENESS,
        "zone": inputs["zone"],
        "calculation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "cpu_cores": inputs["cpu_cores"],
            "gpu_count": inputs["gpu_count"],
            "node_count": inputs["node_count"],
            "partition_name": inputs["partition_name"],
            "memory_gigabytes": inputs["memory_gigabytes"],
            "walltime_hours": inputs["walltime_hours"],
            "resolved_nodelist": inputs.get("resolved_nodelist"),
            "allocated_node_names": inputs.get("allocated_node_names", []),
        },
        "notes": notes,
    }


app = FastAPI(title="Job Prediction API")


@app.get("/")
def root() -> Dict[str, str]:
    """Health and discovery for bare deployment URLs (e.g. opening `/` on Vercel)."""
    return {
        "service": "Job Prediction API",
        "docs": "/docs",
        "predict": "POST /predict",
    }


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
