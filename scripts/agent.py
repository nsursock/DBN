"""Small common interface for MLX RL agents.

The trading bot can depend on this interface without depending on PPO/SAC details.
"""
from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from typing import Any, Dict


class Agent(ABC):
    """Minimal action/persistence interface shared by future trading agents."""

    @abstractmethod
    def predict(self, observation: Any, deterministic: bool = True):
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps: int):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError


class MLXAgent(Agent):
    """Adapter exposing a stable API around an MLX implementation."""

    def __init__(self, model: Any = None):
        self.model = model

    def _resample_csv(self, n: int = 100):
        path = getattr(self, "csv_path", None)
        if not path or not os.path.exists(path):
            return
        if getattr(self, "_csv_file", None):
            self._csv_file.flush()
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = reader.fieldnames
        if len(rows) <= n:
            return
        step = (len(rows) - 1) / (n - 1)
        sampled = [rows[int(i * step)] for i in range(n)]
        base, ext = os.path.splitext(path)
        out = f"{base}_resampled_{n}{ext}"
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled)

    def predict(self, observation: Any, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("MLXAgent has no model")
        return self.model.predict(observation, deterministic=deterministic)

    def learn(self, total_timesteps: int):
        if self.model is None:
            raise RuntimeError("MLXAgent has no model")
        self.model.learn(total_timesteps)
        return self

    def save(self, path: str) -> None:
        if self.model is None or not hasattr(self.model, "save"):
            raise RuntimeError("underlying model cannot be saved")
        self.model.save(path)

    def load(self, path: str):
        if self.model is None or not hasattr(self.model, "load"):
            raise RuntimeError("underlying model cannot be loaded")
        self.model = self.model.load(path)
        return self
