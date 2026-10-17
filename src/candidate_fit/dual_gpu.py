from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request

from candidate_fit.client import _LOCAL_OPENER, require_loopback


def register_client(
    *,
    orchestrator_url: str,
    project_name: str,
    output_path: str | Path,
    github_repo: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "project_name": project_name,
        "output_path": str(Path(output_path).resolve()),
    }
    if github_repo:
        body["github_repo"] = github_repo
    req = request.Request(
        require_loopback(orchestrator_url).rstrip("/") + "/api/clients/register",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _LOCAL_OPENER.open(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict) or not result.get("client_id"):
        raise RuntimeError("dual_gpu_setup returned an invalid registration response")
    return result


def upsert_project(
    *,
    orchestrator_url: str,
    name: str,
    repo_path: str | Path | None = None,
    git_remote: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Attach the repo location to the project so the dual_gpu_setup dashboard can link back to it."""
    body: dict[str, Any] = {"name": name}
    if repo_path:
        body["repo_path"] = str(Path(repo_path).resolve())
    if git_remote:
        body["git_remote"] = git_remote
    req = request.Request(
        require_loopback(orchestrator_url).rstrip("/") + "/api/projects",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _LOCAL_OPENER.open(req, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("dual_gpu_setup returned an invalid project response")
    return result


def save_client_config(path: str | Path, registration: dict[str, Any], orchestrator_url: str) -> None:
    config = {
        "orchestrator_url": orchestrator_url.rstrip("/"),
        "client_id": registration["client_id"],
        "project_name": registration.get("project_name"),
        "metrics_output": registration.get("output_path"),
    }
    Path(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_client_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.exists():
        return {}
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid client configuration: {candidate}")
    return value

