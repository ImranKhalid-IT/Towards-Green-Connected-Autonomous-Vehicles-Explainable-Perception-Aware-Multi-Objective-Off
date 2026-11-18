# GreenCAV — reproducible code

Reference implementation of **"Explainable Edge Offloading for Energy-Efficient Connected Vehicle Perception"**: perception-aware, multi-objective edge offloading with a preference-conditioned PPO policy and uncertainty-gated dual-head XAI feedback.

The whole framework (Eqs. 1–32, Algorithms 1, 2, S1, S2, all baselines, ablations, sweeps and statistics) runs as a **calibrated discrete-event simulator in pure NumPy**. It runs on any CPU and needs no GPU, CARLA or SUMO. Separate modules let you re-measure the detector profiles on real hardware and connect a CARLA–SUMO co-simulation.

## What is reproduced, and what is not

| Part | Status |
|---|---|
| Arithmetic of the paper's tables (Supplementary S2: Wh/h, CO₂e, gains, Cohen's d, expected quality from the model mix, …) | **Exact.** `python scripts/check_paper_tables.py` |
| All models and equations (channel, Markov link, queues, accuracy model, uncertainty, gate, fusion, cost, PPO, GAE, metrics) | Implemented as in the paper |
| Detector profiles (Table 3) | Used exactly as published; `--profile-real` re-measures them on your GPU |
| Algorithms 1, 2, S1, S2 | Implemented (`greencav/algorithms.py`) |
| **The reported numbers of Tables 4–7 and S3–S7** | **Not reproduced number-for-number.** The simulator replaces CARLA–SUMO, BDD100K images and real explainers with synthetic processes (see "Model choices"). The trends can be compared; the absolute values differ. |
| XAI fidelity / deletion / insertion (Table 6) | Functions are provided (`greencav/xai.py`); they need a trained detector and real images |
| CARLA 0.9.15 + SUMO 1.19 co-simulation | Optional bridge in `greencav/cosim/`, **not tested here** (CARLA and SUMO were not available while this code was built) |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # numpy, scipy, pandas, matplotlib, pyyaml
python -m pytest tests                     # or: python tests/test_greencav.py
```

Optional, for real detector profiling: `pip install torch torchvision ultralytics pynvml`.
Optional, for the co-simulation bridge: CARLA 0.9.15 Python API and SUMO 1.19 (TraCI).

## Quick start

```bash
# 1) exact checks of the paper's table arithmetic
python scripts/check_paper_tables.py

# 2) one detector-specific simulation (separate script per detector)
python scripts/sim_yolov5s.py     --runs 30 --tasks 30000
python scripts/sim_yolov8s.py     --runs 30 --tasks 30000
python scripts/sim_yolov8m.py     --runs 30 --tasks 30000
python scripts/sim_faster_rcnn.py --runs 30 --tasks 30000
python scripts/sim_all_detectors.py                 # all four

# 3) full pipeline (training of all variants + evaluation + tables + figures)
python scripts/run_all.py --quick                   # 2e5 steps/variant, 5 runs x 5,000 tasks
python scripts/run_all.py                           # paper budget: 2e6 steps/variant, 30 runs x 30,000 tasks

# train a single variant
python scripts/train.py --variant proposed --steps 2000000
```

Run time on a 2-core CPU: about 0.7 ms per environment step during training, so 2×10⁶ steps take about 20–25 min per variant. The full pipeline (5 variants, 30×30,000 tasks for 6 policies, 6 variants and the sweeps) takes several hours. Use `--workers` to parallelise the evaluation over seeds.

## Code map (paper → code)

| Paper | Code |
|---|---|
| Table 2 (configuration) | `greencav/config.py`, `configs/default.yaml` |
| Table 3 (detectors), Eq. (13) | `greencav/detectors/{yolov5s,yolov8s,yolov8m,faster_rcnn}.py` |
| Eqs. (1)–(2) task release, weather | `greencav/weather.py` |
| Eqs. (3)–(8) noise, rate, Markov link, uplink, modem power, radio energy | `greencav/channel.py` |
| Eqs. (9)–(12) compute, Little's-law queue estimate, edge load | `greencav/detectors/base.py`, `greencav/queueing.py` |
| Eqs. (14)–(21), Algorithm 1 | `greencav/xai.py` |
| Eqs. (22)–(26), Algorithm S1 | `greencav/env.py` (`GreenCAVEnv.step`) |
| Eqs. (27)–(30), Algorithm 2 | `greencav/ppo.py` (`algorithm2_train`) |
| Baselines and ablations | `greencav/policies.py` |
| Eqs. (31)–(32), Algorithm S2, tables | `greencav/evaluate.py`, `greencav/stats.py` |
| Figures 6–10 | `greencav/figures.py` |
| Equations (I1)–(R5) | Used as comparison formulations only. `paper_checks.py` evaluates R5 (the XAI cost). |

## Outputs of `run_all.py`

```
results/
  config.json                         full configuration used
  models/<variant>.npz, *_trainlog.csv
  table4_policy_comparison.{csv,md}   Table 4 analogue
  table5_weather.{csv,md}             Table 5 analogue (incl. model-selection mix)
  table6_xai_overhead.{csv,md}        realised invocation share and amortised cost
  table7_modes_ablation.{csv,md}      Table 7 analogue
  tableS3_uplink / tableS4_traffic / tableS5_vehicle_level / tableS7_statistics
  runs_*.csv                          run-level data (one row per policy x seed)
  figures/*.pdf
```

`tableS7_statistics` reports the paper's normal-approximation interval **and** the paired-difference interval, which needs run-level data (limitation (v) in the paper).

## Model choices of the simulator

These are the values that the paper does not report. All are listed and commented in `greencav/config.py`, and can be changed in the YAML file:

- Pre-processing energy (0.10 J), attributed static power (3 W), and the XAI energy per call (0.30 J on the vehicle, 0.06 J on the edge).
- On-vehicle input buffer (2 frames) and a thermal-throttling model of the vehicle accelerator under sustained 10 Hz load.
- Background RSU traffic as a Markov-modulated Poisson process. Congestion bursts at load 1.25 alternate with normal periods, and the time-average load equals the regime load (25 / 58 / 92 / 98 %).
- Weather persists over consecutive frames (geometric segments, mean 50 tasks) with exact counts n_w = Nϖ_w. The weather estimate on the vehicle is 80 % accurate.
- Synthetic perception signals. u_pred follows a Beta distribution with mean 0.12 / 0.28 / 0.45 / 0.62 by weather, scaled by detector size. Reliability r follows a Beta distribution whose mean falls with A(m, w).
- Cost weights ψ = 2, λ_d = 1, λ_p = 1.5, and E_ref = local Faster R-CNN energy + pre-processing + static power over D.
- PPO: c_V = 0.5, c_H = 0.01, rollout 2048 × 4 environments, α ~ U[0.1, 0.9] during training.
- Local-only and Edge-only use YOLOv8s. The Energy-aware DRL baseline uses a fixed YOLOv8s without weather, uncertainty or XAI inputs.

To use the co-simulation instead of the synthetic processes, drive `GreenCAVEnv` with the weather, frames and density from `greencav/cosim/carla_sumo_bridge.py`. Compute u_pred and r with `xai.predictive_uncertainty` and `xai.alignment_reliability` on the detector outputs.

## Data

Detector training uses BDD100K (https://www.vis.xyz/bdd100k/; Detection 2020 labels, ten classes). The dataset is not redistributed here.

## Example outputs (`examples/quick_run/`)

These outputs come from `python scripts/run_all.py --quick` (2×10⁵ steps per variant, 5 runs × 5,000 tasks, about 11 min on 2 CPU cores). The run is a **smoke test of the pipeline**. The policies are under-trained, so the preference modes still differ little, and five runs cannot give p < 0.0625 in the Wilcoxon test. Use the paper budget (2×10⁶ steps, 30 × 30,000 tasks) before you interpret or report any numbers.
