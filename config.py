"""Typed and validated configuration. Parameters marked assumptions are not supplied by the paper."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import yaml

@dataclass
class Config:
    # Manuscript-known quantities
    seed: int = 2026
    n_tasks: int = 300
    n_vehicles: int = 5
    sensing_hz: float = 10.0
    deadline_ms: float = 100.0
    feature_bytes: int = 25_000
    weather_probs: tuple = (0.87, 0.08, 0.03, 0.02)
    link_rates_mbps: tuple = (5.0, 15.0, 30.0)
    link_overheads_ms: tuple = (5.2, 5.2, 2.3)
    link_success: tuple = (0.979, 0.995, 0.999)
    link_stationary: tuple = (0.05, 0.35, 0.60)
    pre_ms: float = 3.9
    decision_ms: float = 0.5
    return_ms: float = 0.7
    xai_call_ms: float = 15.8
    xai_budget: float = 0.25
    energy_ref_j: float = 2.90
    # The next values are explicitly engineering defaults, not historically validated.
    edge_workers: int = 2
    edge_max_buffer: int = 12
    local_max_buffer: int = 4
    background_edge_rate_hz: float = 30.0
    service_jitter_sigma: float = 0.07
    local_static_w: float = 7.0
    preprocess_energy_j: float = 0.08
    xai_loc_energy_j: float = 0.16
    xai_edge_energy_j: float = 0.12
    fusion_weight: float = 0.45
    accuracy_weight: float = 2.4
    deadline_penalty: float = 2.5
    drop_penalty: float = 4.0
    # Default Markov kernel: P_ij=(1-persist)*stationary[j]+persist*I_ij
    link_persistence: float = 0.84
    alpha: float = 0.5
    enable_xai: bool = True
    feedback_enabled: bool = True
    fixed_detector: int | None = None
    exact_weather_shares: bool = True
    stochastic_training_weather: bool = True
    scenario: str = 'mixed'
    adapter: str = 'synthetic'  # 'synthetic' or 'live-observe'; latter requires external CARLA-SUMO synchronizer.
    carla_host: str = '127.0.0.1'
    carla_port: int = 2000
    traci_port: int = 8813
    ppo_hidden: int = 128
    ppo_lr: float = 3e-4
    ppo_steps: int = 512
    ppo_epochs: int = 5
    ppo_minibatch: int = 128
    ppo_gamma: float = 0.99
    ppo_gae: float = 0.95
    ppo_clip: float = 0.20
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_grad_clip: float = 0.5

    def validate(self) -> None:
        if self.n_tasks < 1 or self.n_vehicles < 1:
            raise ValueError('n_tasks and n_vehicles must be positive')
        if self.deadline_ms <= 0 or self.sensing_hz <= 0 or self.feature_bytes <= 0:
            raise ValueError('Timing and feature size must be positive')
        if abs(sum(self.weather_probs) - 1) > 1e-8 or abs(sum(self.link_stationary)-1)>1e-8:
            raise ValueError('Probability distributions must sum to 1')
        if len(self.weather_probs)!=4 or len(self.link_rates_mbps)!=3 or len(self.link_success)!=3:
            raise ValueError('Unexpected weather/link dimensions')
        if not 0 <= self.alpha <= 1 or not 0 < self.xai_budget < 1:
            raise ValueError('Invalid preference / XAI budget')
        if self.edge_workers < 1 or self.edge_max_buffer < 0 or self.local_max_buffer < 0:
            raise ValueError('Invalid queue configuration')
        if self.fixed_detector is not None and not 0 <= self.fixed_detector < 4:
            raise ValueError('fixed_detector must be 0..3 or null')
        if self.adapter not in ('synthetic', 'live-observe'):
            raise ValueError('Unknown adapter')
        if self.scenario not in ('mixed','clear','light_rain','heavy_rain','dense_smog'):
            raise ValueError('Unknown scenario')

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> 'Config':
        data = {}
        if path:
            data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        unknown = set(data) | set(overrides)
        unknown -= set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('Unknown configuration entries: ' + ', '.join(sorted(unknown)))
        data.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding='utf-8')

    def manifest(self) -> dict:
        return asdict(self) | {'mode': 'modeled synthetic demonstration; NOT independent paper reproduction'}
