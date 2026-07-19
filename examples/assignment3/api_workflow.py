"""Budget-safe client for the Stanford scaling-law training API."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests


class TrainingAPI:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://hyperturing.stanford.edu:8000",
    ):
        self.api_key = api_key or os.environ.get("CS336_API_KEY")
        if not self.api_key:
            raise ValueError("set CS336_API_KEY to your eight-digit Stanford ID")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(
            method, f"{self.base_url}{path}", timeout=30, **kwargs
        )
        if response.status_code == 409:
            raise RuntimeError("duplicate configuration: the API reserved no new budget")
        response.raise_for_status()
        return response.json()

    def budget(self) -> dict[str, float]:
        return self._request("GET", "/budget")

    def experiments(self) -> list[dict[str, Any]]:
        return self._request("GET", "/experiments")

    def experiment(self, experiment_id: int) -> dict[str, Any]:
        return self._request("GET", f"/experiment/{experiment_id}")

    def submit(self, training_config: dict[str, Any]) -> dict[str, Any]:
        reserve = float(training_config["max_runtime_seconds"])
        remaining = float(self.budget()["remaining_seconds"])
        if reserve > remaining:
            raise RuntimeError(
                f"run reserves {reserve:.0f}s but only {remaining:.0f}s remains"
            )
        return self._request("POST", "/submit", json=training_config)

    def wait(self, experiment_id: int, poll_seconds: float = 20) -> dict[str, Any]:
        while True:
            experiment = self.experiment(experiment_id)
            state = experiment["status"]["status_type"]
            if state in {"completed", "failed"}:
                return experiment
            time.sleep(poll_seconds)

    def final_submission(
        self, training_config: dict[str, Any], predicted_final_loss: float
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/final_submission",
            json={
                "training_config": training_config,
                "predicted_final_loss": predicted_final_loss,
            },
        )


def submit_plan(path: Path, execute: bool = False) -> None:
    """Print a plan by default; spend budget only with explicit execute=True."""

    configurations = json.loads(path.read_text(encoding="utf-8"))
    total_reservation = sum(config["max_runtime_seconds"] for config in configurations)
    print(f"{len(configurations)} runs reserve at most {total_reservation / 3600:.2f} B200-hours")
    if not execute:
        print("dry run: inspect the JSON, then call submit_plan(path, execute=True)")
        return
    api = TrainingAPI()
    for config in configurations:
        result = api.submit(config)
        print(result["experiment_id"], result["budget_summary"])
