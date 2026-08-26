"""Small common interface for MLX RL agents.

The trading bot can depend on this interface without depending on PPO/SAC details.
"""
from __future__ import annotations

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
