"""Configuration of the GreenCAV simulator.

Every value that appears in the paper (Table 2: simulation, platform and
learning configuration) is set here with the same number.  Values that the
paper does not report (static power, penalty weights, synthetic-signal
shapes, ...) are marked ``# model choice`` and can be changed in
``configs/default.yaml``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import yaml

WEATHERS: List[str] = ["clear", "light_rain", "heavy_rain", "dense_smog"]
LINK_STATES: List[str] = ["poor", "medium", "good"]


@dataclass
class SimConfig:
    # ---- Table 2: simulation -------------------------------------------------
    n_runs: int = 30                      # independent runs (paired seeds)
    tasks_per_run: int = 30_000           # N
    f_s: float = 10.0                     # perception rate [Hz]
    deadline: float = 0.100               # D [s]
    weather_shares: List[float] = field(default_factory=lambda: [0.87, 0.08, 0.03, 0.02])
    exact_weather_counts: bool = True     # n_w = N * varpi_w exactly (Eq. 2)
    weather_mean_segment: int = 50        # model choice: temporal persistence (tasks)
    weather_classifier_acc: float = 0.80  # model choice: accuracy of on-vehicle weather estimate

    # ---- Table 2: radio -----------------------------------------------------
    carrier_ghz: float = 5.9
    bandwidth_hz: float = 10e6            # B
    p_tx_dbm: float = 20.0                # radiated power
    noise_figure_db: float = 9.0          # NF
    uplink_rates_mbps: List[float] = field(default_factory=lambda: [5.0, 15.0, 30.0])
    uplink_shares: List[float] = field(default_factory=lambda: [0.05, 0.35, 0.60])   # nu
    link_overhead_s: List[float] = field(default_factory=lambda: [0.0052, 0.0052, 0.0023])  # T^ovh_s
    link_success: List[float] = field(default_factory=lambda: [0.979, 0.995, 0.999])         # sigma_s
    radio_energy_j: List[float] = field(default_factory=lambda: [0.093, 0.045, 0.028])       # E^tx per state
    link_mixing: float = 0.20             # model choice: P = (1-e) I + e 1 nu^T
    d_feat_bytes: int = 25_000            # latent upload
    eta_pa: float = 0.30                  # model choice: PA efficiency (only used by channel.modem_power)

    # ---- Table 2: compute / edge -------------------------------------------
    f_loc_ghz: float = 1.8
    edge_workers: int = 4                 # W_e
    edge_buffer: int = 32                 # n_max
    local_buffer: int = 2                 # model choice: frames waiting on the vehicle
    t_pre: float = 0.0039                 # T^pre
    t_dec: float = 0.0005                 # T^dec
    t_ret: float = 0.0007                 # T^ret
    e_pre: float = 0.10                   # model choice: E^pre [J]
    p_static: float = 3.0                 # model choice: P_sta [W]
    compute_jitter: float = 0.10          # model choice: lognormal sigma of inference time
    throttle_gain: float = 1.5            # model choice: thermal slowdown gain
    throttle_onset: float = 0.40          # model choice: utilisation at which throttling starts
    throttle_ema: float = 0.02            # model choice: EMA rate of the thermal state

    # ---- traffic (background load at the RSU) -------------------------------
    traffic_regime: str = "moderate"
    traffic_loads: Dict[str, float] = field(default_factory=lambda: {
        "sparse": 0.25, "moderate": 0.58, "dense": 0.92, "ultra_dense": 0.98})
    background_service_s: float = 0.0141  # model choice: mean service time of other vehicles' tasks
    background_service_cv: float = 0.30   # model choice
    burst_fraction: float = 0.15          # model choice: share of time the RSU is in a congestion burst
    burst_load: float = 1.25              # model choice: offered load during a burst (>1 = overload)
    burst_mean_s: float = 2.0             # model choice: mean burst duration

    # ---- XAI -------------------------------------------------------------------
    xai_budget: float = 0.25              # rho_x
    xai_fusion_weight: float = 0.5        # omega
    xai_call_s: float = 0.0158            # T^xai_call
    e_xai_local: float = 0.30             # model choice: E^xai_loc [J]
    e_xai_edge: float = 0.06              # model choice: E^xai_edge [J]
    xai_enabled: bool = True
    uncertainty_base: List[float] = field(default_factory=lambda: [0.12, 0.28, 0.45, 0.62])
    uncertainty_concentration: float = 12.0   # model choice: Beta concentration

    # ---- cost (Eq. 25) -----------------------------------------------------------
    alpha: float = 0.5
    psi: float = 2.0                      # model choice: quality weight
    lambda_d: float = 1.0                 # model choice: deadline penalty
    lambda_p: float = 1.5                 # model choice: drop penalty

    # ---- vehicle / grid (Eq. 32) --------------------------------------------
    traction_wh_per_km: float = 150.0
    speed_kmh: float = 60.0
    range_km: float = 400.0
    grid_g_per_wh: float = 0.40

    # ---- Table 2: learning ------------------------------------------------------
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    lr: float = 3e-4
    hidden: List[int] = field(default_factory=lambda: [256, 256])
    train_steps: int = 2_000_000
    minibatch: int = 256
    epochs: int = 10
    rollout: int = 2048                   # model choice: T_r
    c_value: float = 0.5                  # model choice: c_V
    c_entropy: float = 0.01               # model choice: c_H
    alpha_train: List[float] = field(default_factory=lambda: [0.1, 0.9])  # model choice: p_alpha = U[a,b]
    train_tasks_per_episode: int = 3000   # model choice

    # ---- ablation switches ------------------------------------------------------
    fixed_detector: Optional[str] = None  # e.g. "YOLOv8s" for the DRL baseline / no-dyn ablation
    use_weather_input: bool = True
    use_xai_input: bool = True
    forced_link_state: Optional[int] = None   # 0/1/2 for the uplink sweep

    def copy(self, **kw) -> "SimConfig":
        c = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(c, k):
                raise KeyError(k)
            setattr(c, k, v)
        return c

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Optional[str] = None, **overrides) -> SimConfig:
    cfg = SimConfig()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cfg.copy(**data)
    return cfg.copy(**overrides) if overrides else cfg
