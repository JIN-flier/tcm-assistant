"""Validated, atomic, local JSON persistence for one user's basic profile."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Medication(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dose: str | None = None
    frequency: str | None = None


class TCMContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sleep: str | None = None
    appetite: str | None = None
    stool: str | None = None
    urination: str | None = None
    cold_heat: str | None = None
    sweating: str | None = None
    tongue: str | None = None
    pulse: str | None = None


class PatientProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    user_id: str = "local-user"
    updated_at: str | None = None
    name: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    biological_sex: Literal["female", "male", "intersex", "unknown"] = "unknown"
    height_cm: float | None = Field(default=None, gt=0, le=300)
    weight_kg: float | None = Field(default=None, gt=0, le=700)
    pregnancy_status: Literal["pregnant", "not_pregnant", "possible", "unknown", "not_applicable"] = "unknown"
    medical_history: list[str] = Field(default_factory=list)
    current_medications: list[Medication] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    tcm_context: TCMContext = Field(default_factory=TCMContext)


DEFAULT_PROFILE = PatientProfile().model_dump(mode="json")


class ProfileStore:
    """Read on every request and replace the JSON file atomically on updates."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> PatientProfile:
        if not self.path.exists():
            profile = PatientProfile.model_validate(deepcopy(DEFAULT_PROFILE))
            self.save(profile, touch_timestamp=False)
            return profile
        with self.path.open(encoding="utf-8") as handle:
            return PatientProfile.model_validate(json.load(handle))

    def save(self, profile: PatientProfile, *, touch_timestamp: bool = True) -> None:
        if touch_timestamp:
            profile = profile.model_copy(
                update={"updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(profile.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def merge(self, patch: dict[str, Any]) -> tuple[PatientProfile, list[str]]:
        current = self.load()
        value = current.model_dump(mode="json")
        changed: list[str] = []
        for key, update in patch.items():
            if key not in value or key in {"schema_version", "user_id", "updated_at"}:
                continue
            if key == "tcm_context" and isinstance(update, dict):
                for nested_key, nested_value in update.items():
                    if (
                        nested_key in value[key]
                        and nested_value is not None
                        and value[key][nested_key] != nested_value
                    ):
                        value[key][nested_key] = nested_value
                        changed.append(f"tcm_context.{nested_key}")
            elif update is not None and value[key] != update:
                value[key] = update
                changed.append(key)
        merged = PatientProfile.model_validate(value)
        if changed:
            self.save(merged)
            merged = self.load()
        return merged, changed
