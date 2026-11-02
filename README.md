# Reproducing the CARLA–SUMO Perception-Aware Offloading Paper

**Paper:** *Towards Green Connected Autonomous Vehicles: Explainable Perception-Aware Multi-Objective Offloading for Real-Time Energy Efficiency*

**Document type:** Implementation, experiment, verification, and result-reproduction guide  
**Source of paper settings:** `main_publication_wording_revised (1).tex` (the supplied manuscript; Tables 1–12 and Algorithms 1–4)  
**Guide updated:** 2026-11-02  
**Status:** **Implementation plan; not an executable/reproduced experiment.**

> **Zaroor parhein:** Ye README batati hai ke paper jaisi simulation aur experiments banane ke liye *kya kya implement, train, run aur verify* karna hoga. Sirf README ya CARLA install karne se paper ke numbers automatically nahin aayenge. Original trained checkpoints, complete source code, scene/dataset definitions, random seeds, and raw per-run measurements is `.tex` upload mein nahin diye gaye. Neeche diya gaya `TARGET` **paper ka reported number hai**, simulation se naya generate kiya hua result **nahin**. Do not tune the simulator to force a match and then describe it as independent validation.

## 0. What we are trying to reproduce

An autonomous vehicle captures an image/sensor frame, decides whether to run object detection locally or on a roadside MEC/edge server, and simultaneously chooses among four detectors. A preference-conditioned PPO agent minimizes system energy, latency, missed deadlines, dropped tasks, and perception error. A lightweight learned saliency/explanation head is invoked on approximately the 25% most uncertain frames; its reliability/uncertainty feedback informs the **next** decision. A CARLA–SUMO co-simulation supplies motion, traffic, weather, and cameras; a calibrated stochastic network/queue emulator supplies V2I and MEC behavior.

**Important boundary:** The paper uses an **effective C-V2X link model**, **conceptual Jetson/MEC compute profiles**, and **specified vehicle/grid conversion inputs**. It does **not** describe a validated physical 5.9 GHz C-V2X radio implementation, a hardware-in-the-loop system, or measured real-world carbon emissions.

### Main reported targets — use for comparison, NOT as preset simulator outputs

| Metric | Proposed policy | Energy-aware DRL baseline | Source |
|---|---:|---:|---|
| Mean end-to-end latency | **49.3 ms** (SD 1.4) | 58.7 ms (SD 1.9) | Table 3 |
| P95 latency | **71.2 ms** | 87.5 ms | Table 3 |
| Deadline violations (not drops) | **0.6%** | 2.6% | Table 3 |
| Task drops | **0.5%** | 1.4% | Table 3 |
| Offloaded tasks | **78.1%** | 72.3% | Table 3 |
| System energy / 30,000 tasks | **38.9 kJ** (SD 1.1) | 45.4 kJ (SD 1.5) | Table 3 |
| System energy / task | **1.30 J** (rounded) | 1.51 J | Table 3 |
| mAP@0.5 | **83.1%** (SD 0.3) | 81.4% (SD 0.4) | Table 3 |
| XAI call / invocation | **15.8 ms / 25%** | n/a | Table 5 |
| Amortized XAI overhead | **3.95 ms/task** | n/a | Tables 5, 11 |
| XAI fidelity | **0.84** | n/a | Table 5 |
| Mean local/vehicle-side workload power | **8.7 Wh per driving hour** | 11.2 Wh/h | Table 10 |
| System workload power | **13.0 Wh per driving hour** | 15.1 Wh/h | Table 10 |
| Attributable CO2e | **0.09 g/km** | 0.10 g/km | Table 10 |

**Reported relative improvements:** latency `(58.7 - 49.3)/58.7 = 16.0%`; energy `(45.4 - 38.9)/45.4 = 14.3%` (rounded). These are *reference targets*. Genuine reproduction may differ and must report the difference.

## 1. Mandatory artifacts and prerequisites

### 1.1 Already available

- [x] Main LaTeX manuscript containing problem definition, equations, algorithms, and summary tables.
- [x] Paper-linked walkthrough video publicly listed on the associated GitHub repository (video **only** observed when this guide was prepared): [repository](https://github.com/ImranKhalid-IT/Towards-Green-Connected-Autonomous-Vehicles-Explainable-Perception-Aware-Multi-Objective-Off).

### 1.2 Need to obtain or create before a full independent reproduction

- [ ] CARLA **0.9.15** binary, compatible Python API, town assets, weather settings and scenario definition.
- [ ] SUMO **1.19.0** binary, traffic routes and synchronized CARLA map; save route seeds and `.sumocfg` files.
- [ ] Object-detection training/evaluation dataset or reproducible CARLA frame + ground-truth export with class mapping; record licensing and train/validation/test split manifests.
- [ ] Four detector model definitions, exact weights, preprocessing/postprocessing, NMS/confidence thresholds, profiling method and reproducible accuracy evaluation.
- [ ] On-vehicle encoder plus **actual 25,000-byte** compressed feature representation, the decoder/edge feature interface, and separate split inference weights for each detector if architectures differ.
- [ ] Trained/teacher explainers and a trained, checkpointed dual-head XAI model, explanation supervision/losses, target masks/box mapping, calibration data, and validation threshold `tau`.
- [ ] PPO agent architecture, hyperparameters, normalization, reward constants, trained checkpoints, training/validation splits, best-checkpoint rule, and preference sampling law.
- [ ] Network Markov transition matrix, channel-state initialization, exact queue scheduling policy, buffer size, parallel stream count, edge-load generation, failure/drop handling, and energy calibrations.
- [ ] 30 **actual** paired evaluation seeds, raw task logs, run-level CSVs, statistical-analysis scripts, and environment manifests.

**Stop/go rule:** When any important input is missing, name the implementation choice and its provenance. It is scientifically invalid to silently treat a newly chosen parameter as the historical one used to produce the manuscript figures.

## 2. Environment / machine setup

**Suggested development environment, not claimed to be the original machine:** Ubuntu 22.04 with NVIDIA GPU, matching driver/CUDA runtime, suitable VRAM and disk, or Windows 10/11 with a compatible CARLA package. CARLA recommends a dedicated GPU equivalent to RTX 2070 or stronger with >=8 GB VRAM and approximately 20 GB disk for the CARLA package; a more powerful GPU is preferable for multi-vehicle camera recording and detector training. **Separate the simulation GPU from the profiled Jetson/RTX 3090 compute budget** if it is not the same hardware; do not mistake host wall-clock time for simulated inference latency.

Required components:

| Component | Version/status |
|---|---|
| CARLA simulator | **0.9.15**, fixed by paper |
| SUMO | **1.19**, fixed by paper; installing the latest SUMO is **not equivalent** |
| Python | Choose version compatible with the pinned CARLA wheel and ML libraries; **record exact version** |
| PyTorch/CUDA | **Not specified in paper**; select compatible versions and lock them |
| Gymnasium or compatible simulator adapter | Implementation dependency, **not specified** |
| PPO implementation (e.g., custom PyTorch or Stable-Baselines3) | Implementation choice; exact policy/reward must match paper |
| Ultralytics YOLO + YOLOv5 environment, torchvision Faster R-CNN | Match detector checkpoints and package versions; these may require separate environments |
| Data libraries | NumPy, pandas, SciPy, pandas/pyarrow, matplotlib, PyYAML, tqdm and logging tools |
| Power / timing instrumentation | Real measuring scripts OR explicitly calibrated profiles; do not conflate the two |

**Example: verify operating environment (these commands are real, but do not install the complete research project):**

```bash
nvidia-smi                 # if an NVIDIA GPU is present
python3 --version
sumo --version             # must report 1.19.x for paper-version reproduction
which sumo
# After installing CARLA's compatible Python wheel:
python3 -c "import carla; print(carla.__file__)"
```

**CARLA 0.9.15 installation:** obtain the release archive and the *matching* CARLA Python wheel. Avoid blindly installing the latest `carla` or `sumo` package. Follow the [CARLA package installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/) and [SUMO installation documentation](https://sumo.dlr.de/docs/Installing/), pinning versions to those in the manuscript. In Python, install and record packages in a lock file; commands such as `pip install torch numpy pandas scipy matplotlib pyyaml` only install general dependencies and are *not* enough to run this research system. A compatible CARLA wheel often comes in the extracted CARLA distribution under `PythonAPI/carla/dist/`; choose the filename matching your actual OS/Python ABI.

**CARLA smoke test (Ubuntu packaged installation):**

```bash
cd /path/to/CARLA_0.9.15
./CarlaUE4.sh             # terminal 1; use headless options only after checking version
```

**SUMO + CARLA smoke test (official co-simulation example, not our experiment):**

```bash
export SUMO_HOME=/path/to/sumo-1.19.0
cd /path/to/CARLA_0.9.15/Co-Simulation/Sumo
python3 run_synchronization.py examples/Town01.sumocfg --sumo-gui
```

The command above assumes the correct CARLA server is already running and the SUMO 1.19 binaries are on `PATH`. The documented co-simulation examples cover Town01, Town04 and Town05. **The paper does not identify its town or traffic routes**; Town01 is only a smoke-test example, **not proof that Town01 produced the paper results**. CARLA co-simulation synchronizes traffic but is not a complete V2I / MEC offloading simulator. The research-specific environment must be written separately.

## 3. Recommended research repository structure — blueprint, not files claimed to exist

```text
reproduction/
  README.md
  paper/main_publication_wording_revised.tex
  config/
    simulation.yaml                  # CARLA/SUMO/map/seed/timestep
    networks.yaml                    # bandwidth, Markov P, loss, modem power
    compute.yaml                     # detector profiles, queues, energy
    ppo.yaml                         # ALL training hyperparameters
    xai.yaml                         # architecture, loss, gate, omega
    experiments.yaml                 # per-experiment overrides
  scenarios/
    map/                             # CARLA town/OpenDRIVE and SUMO net
    routes/                          # 30 seeded route definitions
    weather/                         # four weather presets and labels
  data/
    raw/                             # image frames, bounding boxes, metadata
    splits/train.txt
    splits/val.txt
    splits/test.txt
    calibration/                     # profiled timings/power and derived values
  models/
    detectors/                        # four frozen detector checkpoints
    split_encoders/                   # validated <=25 kB feature encoders
    xai_teacher/
    xai_dual_head/
    ppo_proposed/
    ppo_energy_drl/
  src/
    cosim/bridge.py                   # synchronized CARLA-SUMO driver
    cosim/scenarios.py                # scene generation & weather control
    cosim/sensor_export.py            # sensor frame and ground truth
    perception/inference.py           # local/edge detector wrappers
    perception/split.py               # encoder/payload/decoder, bytes
    perception/metrics.py             # real AP/mAP evaluation
    communication/markov_link.py
    communication/uplink.py
    compute/queues.py
    compute/energy.py
    xai/train_head.py
    xai/gate_and_feedback.py
    rl/env.py                          # state/action/reward transition
    rl/train_ppo.py
    rl/evaluate.py
    rl/baselines.py
    metrics/aggregate.py
    metrics/stats.py
    metrics/plots.py
    tests/                           # analytical & integration tests
  scripts/
    01_smoke_test.sh
    02_generate_scenarios.sh
    03_profile_detectors.sh
    04_train_xai.sh
    05_train_ppo.sh
    06_evaluate_paired.sh
    07_analyze_results.sh
  output/
    raw_tasks/                        # per-policy per-seed records
    runs/                             # metrics for each of 30 runs
    tables/                           # regenerated Tables 2-12
    figures/                          # regenerated PDFs
    checkpoints/                      # training & model history
    provenance/                       # hashes, environment, config, seeds
```

Each name above describes a file that **must be implemented** if not already present. The sample CLI commands in later sections are **proposed interfaces** and will not execute until the corresponding modules/scripts are created.

## 4. Lock the paper's simulator specification

### 4.1 Basic task and environment values (from manuscript)

| Key | Paper setting | What to implement |
|---|---|---|
| `carla_version` | `0.9.15` | Fixed server/client version |
| `sumo_version` | `1.19` | Fixed SUMO version |
| `N_tasks_per_run` | **30,000** | Exactly 30k task-release records **per run** |
| `paired_evaluation_runs` | **30** | Thirty independent paired scenario seeds |
| `sensor_rate_hz` | **10** | A task every `0.1 s` per monitored vehicle |
| `deadline_ms` | **100** | Deadline relative to task release |
| weather `clear` | **87% = 26,100 tasks** | Ground truth weather labels |
| weather `light_rain` | **8% = 2,400** | Weather preset plus camera image effect |
| weather `heavy_rain` | **3% = 900** | Adverse weather |
| weather `dense_smog` | **2% = 600** | Fog/smog approximation and documentation |
| traffic, default | **moderate, 30–50 veh/km** | Density/route matching, *not* simply 30–50 vehicles total |
| radio type | effective V2I, RSU assisted | Synthetic/trace-driven rather than real packet-level C-V2X |
| carrier / bandwidth | `5.9 GHz / 10 MHz` | Link model metadata |
| link rates | `5 / 15 / 30 Mbit/s` | Poor / medium / good |
| link occupancy | `5% / 35% / 60%` | Target stationary distribution |
| RF transmit power | `20 dBm`, maximum `23 dBm` | Radiated, not modem input power |
| receiver noise figure | `9 dB` | Thermal noise calculation |
| raw sensor envelope | `2.5–5.0 MB` | Sensor/preprocessed original; do **not** transmit this in split mode |
| transmitted payload | **25,000 bytes exactly** | Measured serialized feature, including clarified payload conventions |
| local compute profile | Jetson-class / effective 1.8 GHz | Calibrated compute model |
| edge compute profile | Xeon-class CPU + RTX 3090-class GPU | Calibrated compute model |
| driving speed | `60 km/h` | For scenario-level energy conversion, not necessarily every CARLA instant |
| traction energy | `150 Wh/km` | Reference scenario conversion |
| nominal range | `400 km` | Range-loss estimate |
| carbon intensity | `0.40 kg CO2e/kWh` | **Scenario assumption**, not field measurement |

For a single ego producing all 30,000 tasks at 10 Hz, the logical simulation interval is **3,000 seconds (50 minutes)**. If 30,000 tasks are aggregated across many vehicles, the actual total simulation time is **different** and must be specified. The manuscript does not resolve that distinction. A common task release timeline must be defined before running all policies.

### 4.2 Weather generation and paired-scene fairness

- Generate weather as explicit labeled **segments or a seeded process**; choose segment durations and transition schedule, document them and verify the realized per-run split. The manuscript specifies proportions but does **not** specify the temporal transition process.
- Hold traffic routes, actor spawn seeds, weather trajectories and frame-collection protocol constant across evaluated policies. If policy actions do not affect driving behavior, pre-record a shared scenario trace and replay it for each policy, which provides stronger paired comparisons.
- Save camera RGB frames, timestamps, ego ID, frame IDs, weather, per-object class and ground-truth boxes. Include cars, pedestrians and cyclists if those classes are to be evaluated, with a consistent taxonomy.
- **Never** determine the weather state or future network condition for the policy using an oracle variable not present in its stated observable inputs. Use `weather_probability_vector` from an explicit estimator (trained on train split).
- Calibrate the weather mixture only on training/validation; hold the test sequence and seeds out.

### 4.3 SUMO/CARLA synchronization

1. Select and document a specific CARLA town (paper does not say which).
2. Generate an OpenDRIVE-to-SUMO road network with matching coordinate transform; verify lane and spawn alignment.
3. Build the route file and document traffic flows, vehicle category distribution, speed limits, traffic lights and random seed.
4. Choose co-simulation tick (e.g., `0.05 s` is supported by CARLA's SUMO example; **this value is a suggested engineering choice, not paper evidence**). A 10 Hz sensor event can occur every 2 such ticks.
5. Set synchronous mode for CARLA, SUMO, all sensors, and the scenario runner; check that timestamps and actor IDs agree.
6. Place one or more RSUs and define road-segment coverage, uplink estimates, and which vehicles share an MEC server. RSU count, placement, propagation loss and mobility/channel coupling are **not fully stated in paper**.
7. Run 100+ synchronized steps with traffic and sensor data; fail the test if image timestamps, vehicle identity or bounding-box transforms drift.
8. Capture visuals/video independently of the measurement pipeline. A beautiful driving animation **cannot demonstrate energy, latency or mAP metrics** without actual logged inference and policy actions.

## 5. Detector training, profiling and perception accuracy

### 5.1 Four model configurations (Table 2 reference values)

| Detector | Resolution | Parameters | Local inference | Local inference energy | Edge inference | Edge inference energy | Clear-weather mAP@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 640×640 | 7.2 M | 45.2 ms | 0.85 J | 12.4 ms | 0.25 J | 78.4% |
| YOLOv8s | 640×640 | 11.2 M | 52.8 ms | 1.05 J | 14.1 ms | 0.32 J | 81.2% |
| YOLOv8m | 640×640 | 25.9 M | 88.5 ms | 1.62 J | 22.3 ms | 0.58 J | 84.6% |
| Faster R-CNN | 1024×1024 | 41.8 M | 145.0 ms | 2.90 J | 45.6 ms | 1.15 J | 86.1% |

These are **paper's computation-only reference profile**, not values measured while writing this README. No radio, queue, preprocessing, throttling or static platform energy is included in them.

### 5.2 How to get valid detector measurements

1. **Build a real labeled dataset:** choose a documented existing set or export frames/ground truth from CARLA. State dataset size, class list, ground-truth pipeline, frame sampling interval, occlusion/visibility thresholds, weather distribution, and split policy. Split by scenario/route rather than random near-duplicate frames to prevent leakage.
2. **Freeze detector checkpoints:** record model architecture and exact SHA-256 of weights, confidence/NMS thresholds, image resizing/letterboxing, normalization, batch size, CPU/GPU precision, and pre/postprocessing.
3. **Measure accuracy:** for all frames per weather and per detector, run actual inference, match predictions with ground-truth bounding boxes, compute AP at IoU=0.5 per class, and take class mean for mAP@0.5. **Do not assume each frame has its own mAP** and do not call an average of per-task probabilities 'mAP'. Overall mAP should normally be recomputed over all held-out predictions/labels, not merely weighted averages of per-weather mAP values. The paper's weather-weighted numbers are useful summary checks but are not generally an exact reconstruction of dataset-level mAP.
4. **Measure isolated compute:** warm up accelerator; use synchronized timing around the forward pass, fixed batch size, clock/power states and precision, and many repetitions. Report mean, SD, P50/P95, and platform metadata.
5. **Measure energy:** integrate measured device power over each inference if hardware is present; otherwise explicitly label the numbers as a calibrated model and detail the calibration. Distinguish wall-power, GPU dynamic power, total vehicle power and static allocation.
6. **Build detector weather degradation:** create measured `A[detector, weather]` from the held-out data. The paper supplies only clear-weather detector values and overall policy weather losses, **not** the full 4×4 weather-by-detector accuracy matrix. This matrix **cannot be recovered uniquely** from the tables.
7. **Create CSV:** `detector,weather,input_size,checkpoint_sha256,n_frames,n_gt,mAP50,local_ms,edge_ms,local_j,edge_j`.
8. **Validation:** edge/local speedups from Table 2 should be ~3.65, 3.74, 3.97 and 3.18, if calibrated to the supplied profile. Treat a mismatch as a calibration or implementation discrepancy, not an excuse to modify measured observations.

### 5.3 Split inference (essential, often omitted)

Offloading **must transmit the learned latent feature** (`25,000 bytes`), *not* the raw `2.5–5 MB` frame. For a real implementation:

- Choose the split location, supported by every detector, and define local encoder/edge decoder interfaces. A generic encoder may require distillation/training to preserve detection accuracy of all four architectures.
- Log payload size **after actual serialization/compression** and before packet/network overhead. State whether '25,000 B' includes headers, metadata and quantization scales.
- Re-run AP tests for local full-path and split-path inference for each detector; without this, the reported accuracy of split/offloaded inference is unverified.
- Account for encoder cost in `T_pre`/`E_pre`, decoder in edge compute or a separate explicitly counted stage, and any feature compression/decompression. **Never double-count encoder work.**
- Use identical input frames for local and remote comparisons. Return metadata and bounding boxes to the vehicle; specify `T_ret` for the result.
- For a paper-faithful **surrogate simulation**, profiling-derived compute times may be used instead of live inference *only if* the output is explicitly labeled profile-driven and a trace-driven perception accuracy model is documented. Do not represent surrogate mAP as detector-evaluated mAP.

## 6. V2I communication model and verification

The effective-link model in the paper has three states:

| State | Probability | Rate | Pure transfer, 25kB | Overhead | Uplink latency | Link delivery success | Radio energy per offloaded task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poor | 0.05 | 5 Mbit/s | 40.0 ms | 5.2 ms | 45.2 ms | 97.9% | 0.093 J |
| Medium | 0.35 | 15 Mbit/s | 13.33 ms | 5.2 ms | 18.53 ms | 99.5% | 0.045 J |
| Good | 0.60 | 30 Mbit/s | 6.67 ms | 2.3 ms | 8.97 ms | 99.9% | 0.028 J |

**Implement exactly:**

```text
N_total_dBm   = -174 + 10*log10(10_000_000) + 9   # -95 dBm
T_min_sec     = (25_000 * 8) / (R_Mbit_s * 1_000_000)
T_up_sec      = T_min_sec + T_overhead_sec[state]
P_modem_W     = 10**((20-30)/10) / amplifier_efficiency + circuit_power_W[state]
E_radio_J     = P_modem_W * T_up_sec
```

- The manuscript supplies the Markov chain's stationary distribution `nu=[0.05,0.35,0.60]` **but not the actual 3×3 transition matrix `P`**. Choose and document a plausible `P` with `nu P = nu`, or recover the original matrix from the authors. Many different matrices have the same stationary distribution but **different link burstiness and deadline tails**.
- Fix and record the Markov initial distribution, state-transition time unit, correlation with vehicle position/weather, and independent RNG stream. Do **not** draw independent states if you claim Markov temporal correlation.
- Calibrate modem electrical power using amplifier efficiency plus circuit/baseband power; **20 dBm is 0.1 W RF radiated power, not 2–3 W electrical modem draw**.
- Fail/drop events must use `sigma[state]=0.979/0.995/0.999` and be independent of the action policy except via which state and whether offloaded; log the random draw.
- Record delays/losses per task and ensure that features never arrive at the edge faster than transmission time.

**Important weighting:** The reference **0.035 J per offloaded task** can be reconstructed by weighting each state by its probability *and offloading fraction*, then dividing by the overall offload share—not by simply averaging radio energy using only state shares. See Section 15.

## 7. Discrete-event MEC + local task queues

The paper's local and edge inference times are task-service profiles. The **simulator must model actual jobs and contention**, not just add a constant average queue delay.

### Required state and service flow

- Represent each task by `(task_id, vehicle_id, released_at, deadline_at, weather, link_state, selected_detector, location, payload_bytes, arrival_time, service_begin, service_end, completion_time, drop_reason, stage_times, energies)`.
- Each vehicle has a local inference resource and bounded local buffer; the shared MEC has `g` parallel inference streams and a bounded queue `n_max`. **`g` and `n_max` are unspecified by paper**; do not conceal the values chosen.
- At task release, the agent receives **estimated**, not future-realized, queue delay `Tq_hat ≈ n_q * mean_service / g`; actual `Tq = start_service - arrival_to_queue` comes from event scheduling.
- MEC offered load: `lambda_e = 10 * sum_v(mean_offload_fraction_v)` tasks/s; `rho_e = lambda_e * mean_service / g` (a reference approximation, not a replacement for discrete-event simulation).
- The main experiment corresponds to a **moderate traffic** reference of ~58% edge load / **8.4 ms** edge queue delay (Table 9). Service mix, number of vehicles, queues and streams must be calibrated to produce a realistic stochastic distribution, rather than assigning `8.4 ms` to every job.
- A local-only system at 10 Hz may accumulate work and throttle: model those effects explicitly if attempting to match its **112.4 ms** mean vs the much shorter isolated YOLOv5s/YOLOv8s times.
- On buffer overflow, mark `drop_reason=local_buffer` or `edge_buffer`. On failed offloading mark `drop_reason=link_failure`. Keep counts separate.
- Clarify **whether a dropped task consumes radio energy, preprocessing, and partial compute energy**. Work already expended must not disappear. Clarify cancellation of tasks that run beyond their deadlines.
- The paper uses **separate** indicators: a task drop `xi=1`; a deadline violation `v=(1-xi)*I(T>100 ms)`. A late *delivered* task is not automatically a 'drop' under the manuscript definition.

### Reference traffic-density scenarios (Table 9)

| Regime | Density (veh/km) | Load | Queue delay | Mean latency | Non-dropped requests/1,000* | Energy/1,000 | Tasks/J* |
|---|---|---:|---:|---:|---:|---:|---:|
| Sparse | 10–20 | 25% | 2.1 ms | 43.1 ms | 998 | 1,189 J | 0.84 |
| Moderate | 30–50 | 58% | 8.4 ms | 49.3 ms | 995 | 1,297 J | 0.77 |
| Dense | 70–100 | 92% | 24.5 ms | 68.4 ms | 976 | 1,479 J | 0.66 |
| Ultra-dense | 120+ | 98% | 56.8 ms | 121.7 ms | 930 | 2,163 J | 0.43 |

\* The paper calls the count `successes/1000`. Since moderate `995/1000` equals the 0.5% drop rate, **confirm whether 'success' means non-dropped versus delivered before deadline**. The two definitions are different. For the default 0.5% drops and 0.6% late-delivered tasks, on-time non-dropped success would be about **98.9%**, not 99.5%, if the denominators are all released tasks.

## 8. Uncertainty, XAI training, 25% gate and next-step feedback

### 8.1 Detection uncertainty

For each predicted box with a **true K-class distribution**, compute normalized entropy `H(p)/log(K)`, and average across detections. Set uncertainty to `1` if no boxes are predicted. If a YOLO checkpoint returns only class confidence or filtered top-class scores, **explicitly reconstruct/calibrate a valid class probability distribution**; treating raw logits/confidence as probabilities invalidates the formula. Keep pre-NMS/post-NMS choice fixed.

### 8.2 Dual-head learned explainer

The supplied manuscript describes a saliency head `S_hat = ReLU(g_theta(F))` and a box-alignment reliability score `r = saliency_mass_inside_predicted_boxes / total_saliency_mass`. It calls the architecture *dual-head*, but the upload does **not** provide layer-by-layer head design, weight initialization, loss-function weights, teacher training/distillation procedure, teacher mask definition, or a model checkpoint.

**Implementation tasks:**

1. Specify encoder feature tensor shapes for each detector, a head producing saliency, and an additional head/task if 'dual-head' is to be a literal architectural claim. Define any adapters for YOLO and Faster R-CNN.
2. Pick, justify and **label as a new design choice** an explanation teacher/target (e.g., a declared Grad-CAM++/perturbation-based target) with validation images and masks.
3. Train the learned head(s) on train data; select checkpoint using held-out validation only. Prevent leakage from test weather/route scenes.
4. Upsample saliency to image resolution, align predicted boxes, clamp nonnegative mass, apply `eps0` to avoid zero division, and log reliability `r`.
5. Measure on the same held-out examples: confidence-drop fidelity from the manuscript's Eq. (fidelity), deletion-AUC, insertion-AUC; define deletion fraction `q`, number of insertion/deletion steps, masks, fill baseline and numerical integration. Those are **missing in text and need documentation**.
6. Benchmark Grad-CAM++, Score-CAM, Sobol and the learned head at comparable image resolution, batch size and device/precision. Reference targets: fidelity **0.72/0.76/0.81/0.84** and per-call cost **12.5/25.4/38.2/15.8 ms**, respectively; memory **310/420/650/95 MB**. These are paper values, not guaranteed external results.
7. Compute `tau` **on validation**, the 75th percentile of predictive uncertainty; freeze it for test. Count actual test invocations; a fixed validation threshold does **not guarantee exactly 25.0% on shifted test weather**. If a strict 25% *online budget* is required, define a causal, predeclared adaptive quantile mechanism and report it separately.
8. For each successful task, set `x_i = I(u_pred_i >= tau)` and `u_i = (1-omega)*u_pred_i + omega*(1-r_i)` if invoked; otherwise `u_i=u_pred_i`, `r_i=r_previous`. On failure/drops define a causal fallback for feedback (e.g., carry previous state with a `feedback_valid` flag), not a fictional detection output.
9. Feed `(u_i,r_i)` only into the **next** decision `(i+1)`. For multiple vehicles with concurrent tasks, use the **last completed** feedback per vehicle with correct completion timestamps; never allow information from an inference that has not yet finished.
10. Record `omega`, gate threshold, head checkpoint hash, per-call costs, XAI execution node and power (vehicle vs MEC). These are not fully specified in the paper.

### 8.3 XAI cost accounting

- `T_xai_call = 15.8 ms` (reference) **only when invoked**; realized overhead `mean(x_i * 15.8ms)`.
- If XAI is on the MEC, charge its compute and energy to MEC; if local, charge local. Match it to the node where detector features exist.
- `0.25 * 15.8 = 3.95 ms/task` is the manuscript's **amortized arithmetic**, not a real hard deadline guarantee.
- Add XAI latency to the execution path if it is required before final task completion, as Eq. (latency) assumes. Distinguish online XAI in the critical path vs asynchronous feedback if testing an alternative architecture; they have different latency definitions.

## 9. Full per-task latency and energy accounting

Use **seconds internally**, convert only for reports. Track actual service/event timestamps and per-stage energy per task.

### 9.1 Latency formula (paper Eq. latency)

```text
T_i = T_pre + T_dec + x_i*T_xai
    + (1-o_i)*T_local(detector)
    + o_i*(T_uplink + T_edge_queue + T_edge(detector) + T_return)
```

Paper Table 11 reference **averaged across all tasks**:

| Stage | Task-weighted mean contribution |
|---|---:|
| Preprocessing + on-vehicle encoding | 3.90 ms |
| Local inference, weighted by 21.9% local tasks | 11.60 ms |
| V2I upload, weighted by 78.1% offloaded tasks | 11.00 ms |
| Edge queue waiting, weighted | 6.60 ms |
| Edge inference, weighted | 11.00 ms |
| XAI, 25% × 15.8 ms | 3.95 ms |
| Policy decision | 0.50 ms |
| Result return | 0.70 ms |
| **Total** | **49.25 ms ≈ 49.3 ms** |

**Important:** Weighted stage contributions **are not timings to add on every single task**. For example, a local task should not also receive an edge uplink delay. Store all per-task components first, then average each component over the whole 30,000-task run.

**Check/possible gap:** Using Table 8's link-state shares, offload fractions and uplink latencies gives approximately **9.87 ms of uplink per released task** in a simple weighted calculation, versus **11.00 ms** reported in Table 11. An additional timing component, correlations, or unrounded measurements may explain the difference, but the paper does not specify such a reconciliation. **Do not silently insert a 1.13 ms fudge factor**; verify against raw task logs and revise the table/explanation if necessary.

### 9.2 Energy formula (paper Eq. energy)

```text
E_vehicle = E_pre
          + (1-o)*(E_local_detector + x*E_xai_local)
          + o*E_radio
          + P_static*T_total
E_system  = E_vehicle + o*(E_edge_detector + x*E_xai_edge)
```

- Reconcile all terms to a single accounting boundary. A device's idle/static draw must not be counted both in the detector profile and `P_static * T_total`.
- Define energy of dropped/late tasks based on work actually spent. Avoid attributing MEC energy when the link failed before MEC arrival.
- `E_edge` and `E_local` from Table 2 are computation-only reference values, not necessarily whole-server or battery energy.
- The paper introduces `k_c`, edge power, amplifier efficiency, circuit power, `P_static`, `E_pre` and XAI energies **without providing enough raw numeric values to regenerate every total uniquely**. All require calibration files or author-supplied values.
- If all tasks are counted, `energy_kJ = sum(E_system_J) / 1000`; `energy_per_task = sum(E_system_J)/30_000` (include or exclude dropped tasks only according to a clearly stated and consistent rule; the paper describes workload energy per 30k requests).

## 10. Implement the preference-conditioned PPO agent

### 10.1 State vector (paper)

For task `i`:

```text
state_i = [
  weather_probability_vector_i,   # 4 probabilities, not true weather oracle
  uncertainty_feedback_prev,     # previous completed task
  reliability_feedback_prev,     # previous completed task
  estimated_uplink_rate_i,
  estimated_edge_queue_wait_i,
  estimated_edge_utilization_i,
  local_backlog_i,
  alpha, beta
]
```

Use explicit feature normalization and record estimator implementation. Include action masking for unavailable/disconnected MEC only if clearly modeled; such masking is an additional design choice.

### 10.2 Action space

- `o ∈ {local, edge}`;
- `m ∈ {YOLOv5s, YOLOv8s, YOLOv8m, Faster R-CNN}`;
- joint discrete action size **8**. You can implement categorical logits for all 8 actions, or two appropriately linked heads and log their joint probability. Any hard feasibility constraint must be documented.

### 10.3 Cost / reward

```text
q_i    = (1 - drop_i) * accuracy_for_selected_detector_weather / 100
c_i    = alpha * (E_system_i / E_ref)
       + beta  * (T_i / 0.100)
       + psi * (1 - q_i)
       + lambda_deadline * late_i
       + lambda_drop * drop_i
reward = -c_i
beta   = 1 - alpha
```

The paper defines this scalar reward but does **not** specify numerical `psi`, `lambda_deadline`, `lambda_drop`, `E_ref`, `gamma`, GAE `lambda`, PPO clipping `epsilon`, policy/value architecture, optimizer learning rate, minibatch size, rollout length, update epochs, train iterations or preference distribution. These are **critical** and must be recovered or set as transparently labeled choices. A plausible PPO configuration is not automatically the paper's configuration.

If AP/mAP is used as the accuracy term, note that **mAP is dataset-level** and cannot be treated as an exact per-task differentiable detection accuracy. The paper uses a profiled weather-conditioned accuracy **surrogate** for reward; use true labels and genuine mAP only for offline evaluation, with an explicit distinction.

### 10.4 Training order and training discipline

1. Finalize and freeze camera/detector datasets/splits and profile cache.
2. Train/freeze detector models and calibrated split encoders.
3. Train/freeze XAI and calibrate `tau` using validation data.
4. Build deterministic/resettable CARLA–SUMO + link + event-queue environments.
5. Implement all five baselines first; verify known corner cases.
6. Train the *fixed-detector energy-aware DRL baseline* using its declared **restricted state** (without weather, uncertainty and XAI inputs) and its energy-aware reward. Use fair compute/training budgets and identical test traces.
7. Train proposed PPO with multiple alpha preferences sampled from a documented distribution. Suggested tests: alpha=0.2, 0.5, 0.8; **training distribution is not specified**.
8. Use PPO clipped objective, GAE, Adam and entropy/value loss as in paper Algorithms; save checkpoint, optimizer state, normalization parameters, calibration thresholds, configs and seeds.
9. Select checkpoint via validation only; no policy tuning on final 30 test seeds. At evaluation choose **argmax action**, not PPO sampling (paper specifies deterministic evaluation).
10. Evaluate on independent test scenarios, all policies receiving the same external random trace where possible.

### 10.5 Data-leakage and reproducibility tests

- [ ] Current policy state has no future queue/service or future weather values.
- [ ] XAI feedback is only available after the inference actually completed.
- [ ] Separate RNG generators for traffic, weather, network, queue/background load, detectors/training and policy training; saved seed map per run.
- [ ] Train/validation/test frame or route leakage checks pass.
- [ ] Detector weights and input preprocessing stay frozen during policy baseline comparisons (unless explicitly declaring different treatment).
- [ ] Reward normalization and running state statistics are frozen during evaluation.
- [ ] At every run, output 30,000 task records with unique IDs and a clear denominator for every metric.

## 11. Five baselines and ablations — implement before claiming the improvement

**Main six policies (paper Table 3):**

1. **Local-only:** force location=vehicle; choose and explicitly specify detector selection rule so the comparison is well-defined. Never offload.
2. **Edge-only:** force location=MEC for all tasks; choose and specify detector selection. Link failures and full edge queue must be counted.
3. **Greedy:** among candidate actions, minimize predicted end-to-end latency, given only currently observed state estimates.
4. **Deadline-aware:** among actions predicted to finish within 100 ms, minimize predicted system energy. Define fallbacks when none satisfy the predicted deadline.
5. **Energy-aware DRL:** PPO with fixed mid-size detector and restricted inputs, trained fairly. The manuscript says *fixed mid-size* but does **not** name whether this is YOLOv8s or YOLOv8m — resolve explicitly.
6. **Proposed:** PPO choosing location + detector, conditioned on preferences, using weather, previous XAI feedback, link and queue features.

**Mandatory ablations (Table 7 targets):**

| Variant | Latency | Energy | Drops | Violations | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| No XAI feedback/gate | 56.4 ms | 42.1 kJ | 1.8% | 1.9% | 76.5% |
| No energy objective | 48.1 ms | 55.6 kJ | 0.8% | 0.5% | 83.0% |
| No dynamic perception (fixed model) | 52.8 ms | 44.5 kJ | 2.5% | 2.8% | 71.4% |
| Full proposed | **49.3 ms** | **38.9 kJ** | **0.5%** | **0.6%** | **83.1%** |

**For scientifically meaningful ablations**, decide whether each agent is retrained from scratch after removal (preferred for architecture ablations) or only evaluated with a removed input; write which. Keep data, reward, training budget and seed protocol comparable. Removing the energy objective changes reward and warrants retraining.

## 12. Experiments: exact schedule to regenerate Tables 3–12

### Experiment A — model profiling (Table 2)

- Four detectors; isolated local and MEC computation latency/energy with common evaluation images; repeated measurements with device information.
- Export raw timings and energy profiles, model accuracy, and confidence intervals/variance where measured.

### Experiment B — overall six-policy comparison (Table 3)

- **30 independent seeds × 6 policies × 30,000 tasks = 5,400,000 policy-task evaluations** for the main comparison, assuming the entire policy set is evaluated for each seed.
- Default/balanced preference `alpha=0.5, beta=0.5`.
- Same 30 external scenario seeds for each policy; a policy may still change downstream queues, which is part of the causal effect.
- Export per-task CSV/Parquet and 30 run-level rows per policy; then report mean ± sample SD across runs.

### Experiment C — four weather states (Table 4)

- Evaluate clear/light rain/heavy rain/dense smog with exact per-run counts `26100/2400/900/600` **if enforcing paper proportions**.
- Report per-weather actual mAP on ground-truth detections, mean entropy, local/edge ratio, latency, deadline violations, energy. Keep weather and class populations consistent.
- Reference proposed weather mAP: `84.5, 79.2, 68.4, 62.1%`; offload `77, 81, 91, 95%`; mean latency `47.3,55.6,71.5,79.2 ms`.

### Experiment D — XAI (Table 5)

- Same samples, mask fractions, image fill rule and accuracy confidence for all four explainers.
- Report inference XAI wall/CPU/GPU time with synchronized clocks and peak GPU memory; run real fidelity/deletion/insertion tests.
- Show selected fraction and confidence intervals; the 25% reference comes from a validation 75th-percentile gate.

### Experiment E — preference modes (Table 6)

| Mode | Alpha (energy) | Beta (latency) | Reference energy | Mean latency | Deadline violations | mAP |
|---|---:|---:|---:|---:|---:|---:|
| Energy saving | 0.8 | 0.2 | 32.4 kJ | 85.2 ms | 18.6% | 82.1% |
| Balanced | 0.5 | 0.5 | 38.9 kJ | 49.3 ms | 0.6% | 83.1% |
| Latency priority | 0.2 | 0.8 | 51.2 kJ | 38.5 ms | 0.2% | 83.3% |

Run **one trained preference-conditioned agent** with different inputs; do not secretly swap separately trained weights unless explicitly testing separate policies. Report energy-saving's 18.6% violations prominently: mean latency <100 ms is NOT sufficient for real-time compliance.

### Experiment F — ablation (Table 7)

- Three variants + full, paired test scenarios and retraining protocol documented; include input and reward removals.
- Regenerate energy/latency/accuracy/violation/drop differences with uncertainty across runs.

### Experiment G — link sensitivity (Table 8)

| Rate | Offload share target | E2E latency target | Link success target |
|---|---:|---:|---:|
| 5 Mbit/s | 32.0% | 92.0 ms | 97.9% |
| 15 Mbit/s | 68.0% | 59.0 ms | 99.5% |
| 30 Mbit/s | 87.8% | 40.0 ms | 99.9% |

Report **conditional** metric denominators and whether link-state mixes are used or each scenario is forced into one state. The table mixes per-state scenario values with a state-weighted overall row; separate those clearly in results code.

### Experiment H — traffic density/scalability (Table 9)

- Four density regimes (10–20, 30–50, 70–100, 120+ veh/km). Define road length for `veh/km`, edge server placement and background workload.
- Measure realized server utilization, *observed* queue waiting, timeout/late/drop outcomes, useful throughput, and energy; do not prescribe the target queue delays as inputs.
- For ultra-dense 98% load, expected reference ~121.7 ms mean latency, which **fails** 100ms deadline and demonstrates a limitation.

### Experiment I — vehicle-level energy/carbon (Table 10)

Convert energy using 10 Hz, 60 km/h, 150 Wh/km, 400 km, carbon 0.40 kg/kWh. This is a *reference scenario model*, not a battery/range field test.

### Experiment J — time breakdown (Table 11)

Derive by summing measured/logged per-task components. Each task's local/edge branches must be mutually exclusive and XAI only count when invoked.

### Experiment K — significance tests (Table 12)

- For each seed and policy save the **same metrics** into a run-level table.
- Compare paired rows (Proposed − DRL; Proposed − Deadline-aware) using two-sided Wilcoxon signed-rank on nonzero differences; store exact method/tie/zero policy and software version.
- Report 95% **paired-difference** bootstrap confidence interval (resample 30 seed-pairs) or another explicit paired CI. The paper's printed CIs are **independent-group approximations based on summary means/SD**, not true paired CIs; do not represent them as paired.
- Report mean paired difference, SD of differences, CI, p-value, Holm correction if performing multiple planned tests, and clearly labeled effect size. Cohen's pooled-SD `d` in paper is not paired effect size `d_z`.
- p <0.001 **cannot be independently reconstructed from paper means and SDs**; it requires actual paired per-run data.

## 13. Required task logs, run logs and metric definitions

### 13.1 `output/raw_tasks/<policy>/<seed>.parquet` schema (minimum)

```text
seed, policy, task_id, vehicle_id, frame_id, weather, ground_truth_ref,
release_s, absolute_deadline_s, decision_s, completion_s,
link_state, rate_mbps, predicted_rate_mbps, vehicle_queue_length,
edge_queue_length, edge_util_est, edge_util_actual,
offload, detector, payload_bytes,
pre_ms, decision_ms, encoder_ms_included_in_pre,
uplink_ms, edge_queue_ms, local_queue_ms, local_infer_ms,
edge_infer_ms, return_ms, xai_call_ms, total_latency_ms,
local_energy_j, edge_energy_j, radio_energy_j, static_energy_j,
pre_energy_j, xai_local_energy_j, xai_edge_energy_j,
vehicle_energy_j, system_energy_j, energy_at_drop_j,
link_fail, local_overflow, edge_overflow, dropped, late_delivered,
completed_before_deadline, detector_prediction_ref,
uncertainty_pred, xai_invoked, tau, alignment_reliability,
uncertainty_fused, feedback_ready_at_s
```

Fields with unknown values (e.g., completion time after drop) should be `NaN`/null plus explicit `drop_reason`, not artificial zeros included in average latency. **Paper Table 3 appears to report latency including policies with drops but does not specify whether dropped tasks are excluded or given timeout latency.** Select and publish a consistent rule, and show sensitivity to it.

### 13.2 `output/runs/metrics.csv` schema

```text
seed, policy, n_released, n_completed, n_dropped,
mean_latency_ms, p95_latency_ms, deadline_violation_pct,
drop_pct, timely_delivered_pct, offloaded_pct,
system_energy_kj, energy_j_per_released_task,
vehicle_energy_j_per_task, xai_invoked_pct,
mAP50, mean_xai_overhead_ms,
mean_edge_wait_ms, avg_server_utilization
```

**Metric definitions** (state them in any publication or figure caption):

- `drop% = 100*sum(dropped)/n_released`.
- `deadline_violation% = 100*sum((not dropped) AND (latency_ms>100))/n_released` as in the paper's indicator, unless a different denominator is reported and labeled.
- `timely_delivered% = 100*sum((not dropped) AND latency_ms<=100)/n_released`.
- `offload% = 100*sum(offload == 1)/n_released` (actions attempted; specify if a separate *successful-offload* share is also tracked).
- `energy_per_task = sum(all workload energy including failed attempts)/n_released`.
- `mean_latency` and `p95_latency`: define included set of tasks and drop handling before computation. Report both completed-only and drop-penalized variants if ambiguity persists.
- `mAP@0.5`: computed from actual bounding-box predictions and ground-truth labels on held-out frames. Report rules for dropped/late detections (missing prediction = empty prediction vs excluded) and also `mAP_on_completed` vs `mAP_timely` if useful.
- `goodput`: separately distinguish non-dropped tasks and on-time correct/usable tasks; do not silently use a changed success definition.
- `run mean ± SD`: calculate across **30 runs**, not standard deviation of 900,000 pooled tasks as if they were 900,000 independent paired experiments.

## 14. Reference calculations that should pass before training

These are **algebra checks on the manuscript numbers**, not empirical reproduction.

```python
# This snippet uses only Python's standard library.
shares = [0.87, 0.08, 0.03, 0.02]
lat = [47.3, 55.6, 71.5, 79.2]
eng = [1.25, 1.48, 1.75, 1.92]
off = [77.0, 81.0, 91.0, 95.0]
map_weather = [84.5, 79.2, 68.4, 62.1]
weighted = lambda arr: sum(p * x for p, x in zip(shares, arr))
assert round(weighted(lat), 1) == 49.3
assert abs(weighted(eng) * 30_000 / 1000 - 38.904) < 1e-6
assert round(weighted(off), 1) == 78.1
assert round(weighted(map_weather), 1) == 83.1  # summary approximation ONLY
assert abs((0.25 * 15.8) - 3.95) < 1e-10
assert abs(sum([3.9,11.6,11,6.6,11,3.95,0.5,0.7]) - 49.25) < 1e-10
# RF 20 dBm = 0.1 W, but full modem electrical consumption is higher.
assert abs(10**((20-30)/10) - 0.1) < 1e-10
link_share = [0.05, 0.35, 0.60]
link_off = [0.32, 0.68, 0.878]
link_uplink_ms = [45.2, 18.5, 9.0]
link_radio_j = [0.093,0.045,0.028]
all_offload = sum(p*o for p,o in zip(link_share,link_off))
conditional_radio_j = sum(p*o*e for p,o,e in zip(link_share,link_off,link_radio_j)) / all_offload
unconditional_uplink_ms = sum(p*o*t for p,o,t in zip(link_share,link_off,link_uplink_ms))
print('Weather-weighted latency (ms):', weighted(lat))
print('Weather-weighted energy (kJ/30k):', weighted(eng)*30_000/1000)
print('Link-weighted offload (%):', all_offload*100)
print('Radio energy PER OFFLOADED task (J):', conditional_radio_j)  # ~0.0345 -> 0.035
print('Uplink contribution PER RELEASED task (ms):', unconditional_uplink_ms) # ~9.87; investigate vs Table 11's 11
print('Paper CO2e g/km (derived):', (1.2968*10/60)*0.4) # ~0.0865 -> 0.09
```

These checks **must not** be used to synthesize individual random task samples and then publish those samples as if measured. A synthetic dataset fitted to the expected means is only a software/plotting test, not independent evidence.

## 15. Why a reproduction may NOT exactly match 49.3 ms / 38.9 kJ

**Open issues requiring author data, rigorous engineering assumptions, or correction before claiming reproduction:**

| ID | Missing specification / possible inconsistency | What must be decided / tested |
|---|---|---|
| B01 | No downloadable full source/checkpoints observed on paper-linked GitHub repository at preparation time | Publish original implementation or explicitly call this a new reimplementation |
| B02 | No exact CARLA town, actor density generator, route files, camera/sensor setup, simulator step and weather transitions | Version and hash scenario configurations; report all selected values |
| B03 | No dataset, bounding-box class taxonomy, ground-truth protocol, weights, train/test splits | Create/audit labeled frames and held-out evaluation |
| B04 | No per-detector/per-weather `A(m,w)` table | Compute measured 4×4 matrix, not arbitrary fitted guesses |
| B05 | '25 kB latent' without encoder architecture/split layer/quantization/decoder accuracy | Implement and measure true 25,000-byte split inference |
| B06 | Markov stationary distribution known but transition matrix unknown | Recover `P` or disclose new calibrated matrix; check temporal correlations |
| B07 | Missing MEC `g`, buffer `n_max`, detailed scheduling, overflow and load parameters | Record queues and calibrate from unbiased traces |
| B08 | Incomplete energy constants and static power calibration | Export watt measurements or label fitted model inputs |
| B09 | Missing PPO policy shape, hyperparameters, reward penalty weights, training budgets | Recover or fully specify and conduct sensitivity study |
| B10 | XAI teacher, full two-head architecture, loss, omega, evaluation mask fraction and threshold protocol missing | Implement fully, benchmark fidelity and ensure causal feedback |
| B11 | mAP treated as weather-weighted scalar `q_i` in reward and Table 4 summary | Distinguish reward surrogate from genuine dataset-level AP/mAP |
| B12 | Table 8 link mixture predicts ~9.87 ms weighted upload, Table 11 lists 11.00 ms | Reconcile from actual stage logs; no hidden fudge factor |
| B13 | Table 9 'successes' are consistent with non-drop count, not necessarily on-time success | Publish separate drop, late and on-time success rates |
| B14 | Table 12 95% CIs are independent approximation even though tests are paired | Compute valid paired bootstrap CIs with raw seed-level measurements |
| B15 | Dropped task latency, energy, and missing detections' contribution to mAP unclear | Fix denominator and timeout conventions explicitly |
| B16 | 30k tasks at 10 Hz vs many-vehicle workload duration not explicit | Clarify monitored ego/task generator and concurrent background traffic |
| B17 | Dynamic encoder/onboard XAI critical path and edge/local assignment require consistent timing | Measure local/remote split path and draw full stage timeline |
| B18 | All 'benchmark' compute/energy values conceptual/calibrated, not verified hardware | Do not claim hardware validation without actual instrumented tests |

**Acceptance rule:** A version is *faithfully implemented* when all B01–B18 have been addressed, reproducibility artifacts exist, and all model components run. A version is *independently reproduced* only after raw test runs produce consistent metrics with valid uncertainty, irrespective of whether they match the manuscript exactly.

## 16. Suggested command-line workflow (PLACEHOLDERS until code is written)

> **Do not copy/paste these commands expecting them to run today.** They document the CLI that should be developed in the proposed project layout. Only the earlier simulator smoke tests and the pure Python sanity-check snippet are self-contained.

```bash
# Phase 1: sanity checks and scene export
bash scripts/01_smoke_test.sh
bash scripts/02_generate_scenarios.sh
python -m src.cosim.sensor_export --config config/simulation.yaml --split train
python -m src.cosim.sensor_export --config config/simulation.yaml --split val
python -m src.cosim.sensor_export --config config/simulation.yaml --split test

# Phase 2: profile detectors and split inference paths
bash scripts/03_profile_detectors.sh

# Phase 3: train + validate explanation head on the frozen split
bash scripts/04_train_xai.sh

# Phase 4: train/retrain proposed PPO + energy DRL baseline and ablations
bash scripts/05_train_ppo.sh

# Phase 5: genuinely independent 30-seed paired test
bash scripts/06_evaluate_paired.sh

# Phase 6: aggregate all 11 experimental tables / statistical tests / figures
bash scripts/07_analyze_results.sh
```

**Preferred execution in steps** rather than attempting all million-plus tasks before tests pass:

1. `100 tasks`, 1 weather, 1 vehicle, forced local; validate all per-task timestamps and energy.
2. `100 tasks`, forced offload at each of three rates; validate payload bytes, delay and drop logic.
3. `1,000 tasks`, 2 vehicles, one shared MEC; validate queue contention and causal feedback.
4. `1,000 tasks`, four detectors, varied weather; validate bbox/mAP and XAI budget.
5. Small PPO training run with three preferences; verify reward monotonicity and that policies learn beyond random.
6. One complete `30,000-task` run for all 6 policies and all 4 weather categories; inspect distributions, not just mean.
7. Freeze all configs/checkpoints; run the **30 paired test seeds** and log everything.
8. Run ablations/sensitivities without using test output to tune training choices; generate tables/plots and compare with reported values.

## 17. Debugging checklist: if the observed results are far off

### If latency is >100 ms too often

- Confirm **25 kB** upload rather than 2.5–5 MB; check rate units (Mbit/s ≠ MB/s) and seconds/ms conversions.
- Confirm no XAI call on all tasks and that only 25% (or measured gate rate) pay the 15.8ms overhead.
- Check encoder/split compute and GPU synchronization; verify no host simulation frame-time accidentally used as model latency.
- Check MEC queue's observed backlog and concurrent vehicle rate; ensure no hidden service oversubscription.
- Check your detector chosen by PPO; Faster R-CNN local inference alone is 145 ms in Table 2.
- Inspect 95th/99th percentile of delay; a low mean alone can hide drops or deadline misses.

### If energy is too high or too low

- Check 20 dBm conversion gives **0.1 W RF**, but modem electrical power includes PA/circuit consumption.
- Check that `E_pre`, static power, local/edge inference and XAI are each counted once.
- Charge failed uplinks for the radio work used before failure; do not charge MEC for jobs never received.
- Make sure total is **vehicle + radio + MEC**, not vehicle-only.
- If simulating with calibrated values, document calibration residuals and verify each stage against Table 2/8.

### If mAP is much lower or higher

- Check dataset, class mapping, GT export and CARLA occlusion; use consistent detector weights and thresholds.
- Check full-vs-split detections produce similar predictions if they are claimed comparable.
- Recalculate **real AP from boxes**; do not assign paper's 83.1% as each task's accuracy.
- Verify weather effects are measured on held-out frames, not manually forced via `A(m,w)` and counted as independent measured mAP.
- Verify how dropped/late frames count toward accuracy; report all variants explicitly.

### If results appear implausibly perfect

- Check policy isn't reading **future** rate/queue/weather or true detector correctness as input.
- Check test data not reused in train or gate calibration.
- Check evaluation uses independent test seeds and truly paired exogenous traces.
- Check targets are not assigned directly as simulator outputs or hard-coded by policy.

## 18. Output files required for a reproducible paper submission

At the end, the repo should contain:

- [ ] Complete source code, fixed requirements/lock file, setup instructions and license/third-party attribution.
- [ ] Dataset access instructions or export code and class mapping, exact train/val/test manifests and checksums.
- [ ] CARLA town, SUMO network/routes, four weather presets, RSU coordinates, sensor setup and 30 external seeds.
- [ ] 4 detector checkpoints or verifiable third-party hashes; encoder/decoder and XAI checkpoints.
- [ ] PPO model and normalization checkpoints, all hyperparameters, reward weights and training curves.
- [ ] `output/raw_tasks/` with task traces for all policies/seeds, detector predictions/ground truth, profiling data and failure reasons.
- [ ] `output/runs/metrics.csv` with 30 runs per comparison (and clearly separate ablations/sensitivity experiments).
- [ ] Recomputed Tables **2 through 12**, generated from the recorded data rather than copied into plotting scripts.
- [ ] Paired Wilcoxon analysis, paired confidence intervals, effect sizes, multiple-test policy, statistical code and environment versions.
- [ ] All figure PDFs; optional rendered demo video explicitly identified as illustration, not simulation evidence.
- [ ] Transparent `deviations_from_manuscript.md` documenting any mismatched settings or results.
- [ ] A repeatable `make reproduce` or scripted pipeline that checks prerequisites and fails with useful messages when an artifact is missing.

## 19. Reproduction completion / acceptance checklist

**A. Setup completed**

- [ ] CARLA 0.9.15 and SUMO 1.19 versions verified.
- [ ] Map/route/tick/sensors/RSU/vehicles/weather scripts committed and timed synchronization verified.
- [ ] 30 paired external test seeds and training/validation/test isolation recorded.

**B. Perception and networking completed**

- [ ] Four detectors run and emit actual bounding boxes; isolated AP, latency and power audited.
- [ ] Encoder produces validated 25,000 B payload and edge can decode and run detector.
- [ ] Markov link satisfies stationary distribution and transfer/failure model validation tests.
- [ ] Local/MEC queue and complete energy ledger pass conservation/timestamp sanity checks.

**C. Learning completed**

- [ ] XAI head training and fidelity protocol finalized; gate independently calibrated.
- [ ] Proposed PPO with 8 actions and two preference inputs trained from declared split.
- [ ] Five baselines and three ablations implemented with fair budgets.

**D. Paper's metrics completed**

- [ ] Raw per-task + per-run metrics exist and are independently aggregateable.
- [ ] Mean, P95, violations, drops, offload, energy and mAP definitions are fixed.
- [ ] All figures and tables are regenerated from logs.
- [ ] Wilcoxon and confidence intervals use **actual** 30 paired run differences.
- [ ] Differences from reported 49.3ms / 38.9kJ / 83.1% are documented, not hidden.

## 20. Recommended priority order (Roman Urdu)

**Step 1:** CARLA 0.9.15 + SUMO 1.19 install karo; synchronization aur moving cars ke saath camera frame + GT record karo.  
**Step 2:** Four detectors ko exact, documented test dataset par run karo; real bounding-box mAP aur local/edge compute profile hasil karo.  
**Step 3:** 25,000-byte feature encoder/decoder aur 3-state link model banao; upload time verify karo.  
**Step 4:** Multi-vehicle edge queue, local queue, latency aur full energy accounting implement karo.  
**Step 5:** Dual-head XAI train karo; validation se `tau` set karo aur feedback next completed perception decision tak pahunchao.  
**Step 6:** Reward aur joint eight-action PPO train karo; alag fair Energy-aware DRL baseline bhi train karo.  
**Step 7:** Main six policies, three ablations, weather, preference, link aur traffic-density experiments run karo.  
**Step 8:** **30 paired test seeds × 30k tasks** ke logs se tables, charts, actual mAP, Wilcoxon/CI generate karo.  
**Step 9:** Paper ke reported targets se compare karo. Jo difference ho woh honestly report karo; assumptions, code, dataset aur checkpoints archive karo.

**Bottom line:** Manuscript ke target metrics ko *verify* karne ke liye **working CARLA–SUMO scenarios + actual detectors + 25 kB split feature + stochastic V2I/MEC simulator + trained XAI + trained PPO + 30 paired raw test runs** zaroori hain. Manuscript ke aggregate tables alone se complete original experiment exactly reconstruct nahi kiya ja sakta.

## References and verified setup links

1. CARLA installation: https://carla.readthedocs.io/en/latest/start_quickstart/
2. CARLA–SUMO co-simulation guide: https://carla.readthedocs.io/en/latest/adv_sumo/
3. SUMO installation instructions: https://sumo.dlr.de/docs/Installing/
4. SUMO TraCI interface: https://sumo.dlr.de/docs/TraCI/
5. Paper-associated GitHub walkthrough (as linked in the provided `.tex`): https://github.com/ImranKhalid-IT/Towards-Green-Connected-Autonomous-Vehicles-Explainable-Perception-Aware-Multi-Objective-Off

---

**Reproduction integrity notice:** A simulator adjusted to yield the *same* aggregate numbers is not by itself evidence that the findings were independently reproduced. Report the real data, code, assumptions, task samples, seeds, and variance; distinguish the authors' reported figures from newly measured ones.

