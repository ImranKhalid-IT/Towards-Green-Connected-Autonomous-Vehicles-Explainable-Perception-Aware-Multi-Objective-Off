# GreenCAV: Real-World Perception and CARLA–SUMO Co-Simulation Framework

**Project focus:** BDD100K real-world object detection, CARLA 0.9.15 and SUMO 1.19 co-simulation, cloud-GPU model training, and the experimental components required to reproduce a perception-aware computation-offloading framework for connected autonomous vehicles.

---

## 1. Project Overview

GreenCAV is a modular research implementation scaffold for studying energy-efficient, latency-aware computation offloading in connected autonomous vehicles (CAVs). The intended framework combines:

- Real-world object-detection data from **BDD100K**.
- Controlled traffic, vehicle-motion, camera, and weather scenarios generated with **CARLA 0.9.15** and **SUMO 1.19**.
- Four object-detection models: **YOLOv5s, YOLOv8s, YOLOv8m, and Faster R-CNN**.
- A model of vehicle-to-infrastructure (V2I) communication and multi-access edge computing (MEC) queues.
- A preference-conditioned Proximal Policy Optimization (PPO) agent for joint execution-location and detector selection.
- An uncertainty-gated, dual-head explainable-AI (XAI) component whose feedback is used by subsequent policy decisions.
- Per-task latency, energy, deadline, drop, perception-accuracy, and statistical analyses.

**Important scope statement:** The current Python package implements dataset preparation, selected detector-training and evaluation workflows, latency profiling, weather-manifest export, a CARLA data collector, a launcher for CARLA's official SUMO synchronization script, and result/provenance utilities. It is an implementation scaffold—not yet a complete, independently validated reproduction of every component in the manuscript. In particular, the complete V2I/MEC simulator, split-inference encoder/decoder, trained dual-head XAI model, PPO agent, baselines, and full 30-seed evaluation pipeline still require implementation and validation.

## 2. Scientific and Reproducibility Principles

Read this section before running experiments.

1. **Training and validation separation:** The official BDD100K training annotations are divided deterministically into a training partition and an internal validation partition. The default internal validation fraction is 10%, with seed 42.
2. **Final benchmark isolation:** The official labelled BDD100K validation partition is reserved as the final benchmark. Do not repeatedly use it for hyperparameter tuning or model selection.
3. **Official test data:** Do not calculate mAP on the official BDD100K test images unless valid labels are available to the research team under the dataset's access and licence terms.
4. **Separate real and synthetic domains:** CARLA-generated data are synthetic. Report CARLA results and BDD100K results separately. Do not combine their AP or mAP values into a single score without a clearly justified and documented evaluation protocol.
5. **CARLA annotation limitations:** The current CARLA collector labels only a subset of the BDD100K categories. It is not a complete 10-class BDD100K annotation pipeline. Visually inspect and validate projected bounding boxes before treating them as ground truth.
6. **Official SUMO synchronization:** The SUMO launcher delegates to the synchronization script supplied with the relevant CARLA release. It does not implement an independent actor-mirroring or synchronization system.
7. **Hardware-energy boundaries:** Cloud-GPU inference measurements are not measurements of vehicle-side Jetson energy or MEC energy. Use the actual target hardware or calibrated power instrumentation for system-energy claims.
8. **No fabricated results:** Reference values such as 83.1% mAP, 49.3 ms latency, and 38.9 kJ energy are not hard-coded as experimental outputs. Report values produced by actual measurements or clearly labelled calibrated models.
9. **Record deviations:** If a manuscript parameter is unavailable, document the selected implementation value, its rationale, and its provenance. Do not present a newly selected value as though it were the original historical setting.
10. **Distinguish simulation from reproduction:** A simulator tuned to match published aggregate values is not, by itself, evidence of independent reproduction. Preserve raw data, source code, configurations, seeds, model checkpoints, and uncertainty estimates.

---

## 3. Reported Reference Results

The following values are reported in the manuscript and are provided for comparison. They are **reference targets, not guaranteed outputs of this codebase and not preset simulator values**.

| Metric | Proposed policy | Energy-aware DRL baseline |
|---|---:|---:|
| Mean end-to-end latency | **49.3 ms** (SD 1.4) | 58.7 ms (SD 1.9) |
| P95 latency | **71.2 ms** | 87.5 ms |
| Deadline violations (excluding dropped tasks) | **0.6%** | 2.6% |
| Task-drop rate | **0.5%** | 1.4% |
| Offloaded tasks | **78.1%** | 72.3% |
| System energy per 30,000 tasks | **38.9 kJ** (SD 1.1) | 45.4 kJ (SD 1.5) |
| System energy per task | **1.30 J** | 1.51 J |
| mAP@0.5 | **83.1%** (SD 0.3) | 81.4% (SD 0.4) |
| XAI execution time per invocation | **15.8 ms** | Not applicable |
| XAI invocation rate | **25%** | Not applicable |
| Amortized XAI overhead | **3.95 ms/task** | Not applicable |
| XAI fidelity | **0.84** | Not applicable |
| Vehicle-side workload energy rate | **8.7 Wh per driving hour** | 11.2 Wh per driving hour |
| System workload energy rate | **13.0 Wh per driving hour** | 15.1 Wh per driving hour |
| Attributable CO₂e | **0.09 g/km** | 0.10 g/km |

The manuscript reports relative reductions of approximately **16.0% in mean latency** and **14.3% in system energy**:

- Latency reduction: `(58.7 - 49.3) / 58.7 = 16.0%`.
- Energy reduction: `(45.4 - 38.9) / 45.4 = 14.3%`.

These are manuscript-reported comparisons. A genuine reproduction may produce different values; any differences must be reported rather than concealed.

### Important modelling boundary

The manuscript uses an **effective C-V2X link model**, conceptual Jetson/MEC compute profiles, and specified vehicle/grid conversion inputs. It does not establish a validated physical 5.9 GHz C-V2X radio implementation, a hardware-in-the-loop system, or field-measured vehicle carbon emissions. Do not describe the model as a physical radio or hardware-in-the-loop validation unless those experiments have actually been performed.

---

## 4. Current Package: Implemented Components

The files below are present in this project package.

| File | Current purpose |
|---|---|
| `greencav/bdd_converter.py` | Converts BDD100K Detection 2020 annotations to YOLO format and creates the deterministic train/internal-validation split. |
| `greencav/classes.py` | Defines the 10 BDD100K detection classes and category-to-ID mapping. |
| `greencav/yolo_train.py` | Trains YOLOv8s and YOLOv8m and evaluates YOLO models. |
| `greencav/fasterrcnn.py` | Provides a BDD100K dataset adapter, Faster R-CNN training, and COCO-style evaluation. |
| `greencav/profile.py` | Measures repeated YOLO inference latency on a cloud GPU or CPU. |
| `greencav/weather.py` | Exports BDD100K weather-specific image manifests and sample counts. It does not calculate weather-specific AP. |
| `greencav/carla_collect.py` | Collects CARLA RGB frames and projected 2D boxes for supported actor types. |
| `greencav/official_bridge.py` | Launches the official CARLA–SUMO synchronization script supplied with the CARLA installation. |
| `greencav/empirical_profiles.py` | Combines available detector evaluation and profiling results into empirical profile records. Energy remains unset unless measured separately. |
| `greencav/provenance.py` | Records Python, platform, PyTorch, CUDA, GPU, Ultralytics, and NVIDIA system information where available. |
| `greencav/results.py` | Merges available evaluation and profiling JSON files into a CSV summary. |

### Included command-line scripts

- `scripts/prepare_bdd100k.py`
- `scripts/train_yolo.py`
- `scripts/evaluate_yolo.py`
- `scripts/train_yolov5s.sh`
- `scripts/train_fasterrcnn.py`
- `scripts/evaluate_fasterrcnn.py`
- `scripts/profile_detector.py`
- `scripts/export_weather.py`
- `scripts/collect_carla.py`
- `scripts/run_official_sumo_bridge.py`
- `scripts/build_empirical_profiles.py`
- `scripts/merge_results.py`
- `scripts/run_all_detectors.sh`
- `scripts/run_all_evaluations.sh`

### Components still required for full manuscript reproduction

The current package does **not** yet provide a complete implementation of:

- The full stochastic V2I Markov-link model and its transition matrix.
- The discrete-event local and MEC queue simulator.
- The 25,000-byte split-inference encoder, serializer, decoder, and validated end-to-end split path.
- The complete weather-conditioned detector-accuracy model.
- The trained dual-head XAI architecture, teacher, training losses, and checkpoint.
- The uncertainty-gating and causal feedback mechanism.
- The preference-conditioned PPO environment, policy, training procedure, and checkpoints.
- All five baselines, the three ablations, and the complete experiment runner.
- The complete 30-seed paired statistical-analysis pipeline.
- Target-device energy measurement and calibration.

The later sections of this README specify the research protocol and the additional work needed to implement these components. Proposed commands and directory structures are identified as such; they are not claims that the corresponding code already exists.

---

## 5. Environment and Installation

### 5.1 Recommended development environment

A practical development environment is Ubuntu 22.04 with an NVIDIA GPU, a compatible driver/CUDA runtime, sufficient VRAM, and adequate disk space. Windows 10/11 may also be used if the selected CARLA release and Python API are compatible.

For 640 × 640 detector training, a GPU with approximately 24 GB VRAM provides useful headroom. Smaller GPUs may require smaller batch sizes or image sizes. CARLA itself also benefits from a dedicated GPU; the CARLA package documentation recommends a GPU comparable to an RTX 2070 or better, with at least 8 GB VRAM. These are practical recommendations, not guarantees of performance.

Where possible, separate the machine running CARLA/SUMO from the hardware whose inference time or energy is being profiled. Do not treat host simulation wall-clock time as detector inference latency.

### 5.2 Required software

| Component | Version or status |
|---|---|
| CARLA | **0.9.15** for manuscript-version compatibility |
| SUMO | **1.19.x** for manuscript-version compatibility |
| Python | Select a version compatible with the CARLA wheel and ML libraries; record the exact version. |
| PyTorch and CUDA | Not specified in the manuscript. Select compatible versions and lock them. |
| Ultralytics | Pin the version used for YOLOv8 experiments. |
| YOLOv5 | Use the official YOLOv5 repository and record its Git commit. |
| torchvision | Use a version compatible with the selected PyTorch release. |
| Gymnasium or simulator adapter | Required for a future RL environment; not specified in the manuscript. |
| PPO implementation | Custom PyTorch or a compatible RL library; the exact implementation must be documented. |
| Data and analysis tools | NumPy, pandas, SciPy, pyarrow, matplotlib, PyYAML, tqdm, and logging utilities. |
| Power/timing instrumentation | Use actual measurement scripts or explicitly calibrated profiles. Do not conflate the two. |

### 5.3 Create a Python environment

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[all]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The `.[all]` extra installs the project's listed ML and utility dependencies. Select a PyTorch build compatible with the NVIDIA driver and CUDA environment. Depending on the operating system and Python version, CARLA may require installing the matching wheel supplied in the CARLA distribution rather than installing an arbitrary package from PyPI.

For CARLA, install the Python API that matches **CARLA 0.9.15**, the operating system, and the Python ABI. Do not blindly install the latest CARLA or SUMO release.

### 5.4 Record the software and hardware environment

```bash
python -m greencav.provenance
mkdir -p outputs
pip freeze > outputs/requirements-lock.txt

nvidia-smi
python --version
sumo --version
```

Record the GPU model, VRAM, driver, CUDA version, PyTorch version, Ultralytics version, Python version, operating system, and cloud VM type. Keep the environment manifest with every experiment.

---

## 6. BDD100K Dataset Preparation

### 6.1 Download the dataset

Download the official **BDD100K 100K Images** and **Detection 2020 Labels** from the dataset's official distribution pages:

- [BDD100K dataset](https://www.vis.xyz/bdd100k/)
- [BDD100K documentation and downloads](https://doc.bdd100k.com/download.html)

Follow the dataset's access conditions and licence. This project does not bypass the official download process.

### 6.2 Expected directory structure

Place the downloaded files in the following structure:

```text
data/
└── bdd100k/
    ├── images/
    │   └── 100k/
    │       ├── train/
    │       ├── val/
    │       └── test/
    └── labels/
        ├── bdd100k_labels_images_train.json
        └── bdd100k_labels_images_val.json
```

The converter expects the official train and labelled validation images and JSON files at these paths.

### 6.3 Convert annotations and create the split

Run:

```bash
python scripts/prepare_bdd100k.py
```

By default, the converter:

1. Reads the official BDD100K training annotations.
2. Uses seed `42` to create a deterministic split.
3. Assigns 90% of the official training records to `train`.
4. Assigns 10% to `val_internal`.
5. Converts the official labelled validation partition to `benchmark`.
6. Writes YOLO-format labels and a split manifest.

The output structure is:

```text
data/bdd100k_yolo/
├── images/
│   ├── train/
│   ├── val_internal/
│   └── benchmark/
├── labels/
│   ├── train/
│   ├── val_internal/
│   └── benchmark/
└── split_manifest.json
```

The `split_manifest.json` file records the split seed, record counts, and conversion reports.

### 6.4 Required annotation checks

Before a full training run:

- Overlay bounding boxes on at least 100 randomly selected images.
- Verify class IDs, coordinates, image dimensions, and category names.
- Check missing images, empty labels, invalid boxes, and duplicate records.
- Confirm that no image or scene leaks between the training, internal-validation, and final benchmark partitions.
- Preserve the split manifest and dataset checksums.

The official labelled validation partition is reserved for final benchmarking. Use `val_internal` for model selection and tuning.

### 6.5 BDD100K classes

The project uses the following 10 classes:

1. Pedestrian
2. Rider
3. Car
4. Truck
5. Bus
6. Train
7. Motorcycle
8. Bicycle
9. Traffic light
10. Traffic sign

The category mapping is defined in `greencav/classes.py`. Check the mapping against the downloaded annotation schema before training.

---

## 7. Detector Training and Evaluation

### 7.1 YOLOv8s and YOLOv8m

Train YOLOv8s:

```bash
python scripts/train_yolo.py \
  --model yolov8s \
  --epochs 100 \
  --batch 16 \
  --device 0
```

Train YOLOv8m:

```bash
python scripts/train_yolo.py \
  --model yolov8m \
  --epochs 100 \
  --batch 8 \
  --device 0
```

Adjust the batch size to the available GPU memory. The training script uses the internal validation partition specified in `configs/bdd100k_train.yaml`.

### 7.2 Exact YOLOv5s

The project includes `scripts/train_yolov5s.sh`, which clones the official YOLOv5 repository if it is not already present and trains the `yolov5s.pt` model.

```bash
bash scripts/train_yolov5s.sh
```

Record the Git commit printed by the script and preserve it in the experiment manifest. Do not describe `yolov5su.pt` as the original YOLOv5s checkpoint; it belongs to a different Ultralytics YOLOv5u model family.

### 7.3 Faster R-CNN

Train Faster R-CNN:

```bash
python scripts/train_fasterrcnn.py \
  --epochs 20 \
  --batch 4 \
  --device cuda
```

Evaluate the saved checkpoint:

```bash
python scripts/evaluate_fasterrcnn.py \
  --checkpoint checkpoints/fasterrcnn_bdd100k.pth \
  --device cuda
```

The implementation uses a Faster R-CNN ResNet50-FPN detector and COCO-style bounding-box evaluation at IoU thresholds from 0.50 to 0.95, including AP50. Validate the evaluator on a small, manually inspected subset before relying on its output.

### 7.4 Final BDD100K benchmark evaluation

After model selection is complete, evaluate the selected YOLOv8 checkpoints on the official labelled validation benchmark:

```bash
python scripts/evaluate_yolo.py \
  --weights runs/bdd100k/yolov8s/weights/best.pt

python scripts/evaluate_yolo.py \
  --weights runs/bdd100k/yolov8m/weights/best.pt
```

The evaluation script uses `configs/bdd100k_benchmark.yaml` by default. YOLOv5 evaluation should be performed with the same pinned official YOLOv5 repository and its validation command using `--task val`.

Keep the following consistent across model comparisons:

- Dataset split and class taxonomy.
- Image size and resizing/letterboxing procedure.
- Confidence and NMS thresholds.
- Precision mode and inference batch size.
- Evaluation implementation and metric definitions.

Do not compare results produced with materially different evaluation settings without explaining the differences.

### 7.5 Current detector reference profiles from the manuscript

The following values are manuscript-reported computation-only profiles. They are **not measurements produced by this README or guaranteed outputs of the code**.

| Detector | Input resolution | Parameters | Local inference | Local energy | Edge inference | Edge energy | Clear-weather mAP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5s | 640 × 640 | 7.2 M | 45.2 ms | 0.85 J | 12.4 ms | 0.25 J | 78.4% |
| YOLOv8s | 640 × 640 | 11.2 M | 52.8 ms | 1.05 J | 14.1 ms | 0.32 J | 81.2% |
| YOLOv8m | 640 × 640 | 25.9 M | 88.5 ms | 1.62 J | 22.3 ms | 0.58 J | 84.6% |
| Faster R-CNN | 1024 × 1024 | 41.8 M | 145.0 ms | 2.90 J | 45.6 ms | 1.15 J | 86.1% |

These values exclude radio transmission, queue waiting, preprocessing, throttling, and static platform energy.

### 7.6 Requirements for valid detector measurements

For a publishable comparison:

1. Use a documented labelled dataset or a reproducible CARLA frame and ground-truth export.
2. Record dataset size, class list, frame-sampling interval, occlusion/visibility rules, weather distribution, and split policy.
3. Split by scenario or route where possible, rather than randomly splitting near-duplicate video frames.
4. Freeze each detector checkpoint and record its SHA-256 hash, preprocessing, confidence threshold, NMS threshold, precision, and batch size.
5. Run actual inference on held-out frames and match predicted boxes to ground-truth boxes.
6. Compute AP at IoU 0.50 for each class, then calculate mAP@0.5 across classes.
7. Report mean, standard deviation, P50, and P95 inference latency, together with hardware and software metadata.
8. Measure device power where possible. Otherwise, label energy as a calibrated model and describe the calibration procedure.
9. Distinguish GPU dynamic power, full-device wall power, total vehicle power, and static power allocation.
10. Store the measurements in a machine-readable file.

A useful detector-profile CSV schema is:

```text
detector,weather,input_size,checkpoint_sha256,n_frames,n_gt,
mAP50,local_ms,edge_ms,local_j,edge_j
```

The manuscript provides clear-weather detector values and overall policy weather results, but it does not provide the complete 4 × 4 detector-by-weather accuracy matrix `A[detector, weather]`. That matrix must be measured; it cannot be uniquely recovered from the published summary tables.

The edge/local speedups implied by the reference profiles are approximately 3.65, 3.74, 3.97, and 3.18 for YOLOv5s, YOLOv8s, YOLOv8m, and Faster R-CNN, respectively. Treat deviations as calibration or implementation findings; do not alter measured observations to force agreement.

---

## 8. Weather-Stratified Evaluation

The manuscript specifies four weather conditions:

| Weather condition | Share | Tasks per 30,000-task run |
|---|---:|---:|
| Clear | 87% | 26,100 |
| Light rain | 8% | 2,400 |
| Heavy rain | 3% | 900 |
| Dense smog | 2% | 600 |

The current CARLA collector exposes three coarse presets: `clear`, `rain`, and `fog`. These presets are **not yet a one-to-one implementation** of the manuscript's four weather conditions. Define and document the mapping or extend the collector before claiming that the four manuscript conditions have been reproduced.

### 8.1 Export BDD100K weather manifests

Run:

```bash
python scripts/export_weather.py \
  --labels data/bdd100k/labels/bdd100k_labels_images_val.json \
  --images data/bdd100k/images/100k/val
```

This script exports image manifests grouped by the weather metadata in the BDD100K annotations and records the number of images in each group. It does **not** calculate AP or mAP.

To report AP by weather, run each detector on each manifest and evaluate predictions against the corresponding ground truth with one fixed evaluator. Report the sample count for every weather group.

### 8.2 Weather protocol

- Define weather as explicit labelled segments or a seeded stochastic process.
- Document segment durations and transition rules. The manuscript gives proportions but does not specify the temporal transition process.
- Keep traffic routes, actor-spawn seeds, weather trajectories, and frame-collection rules consistent across policies.
- Save RGB frames, timestamps, ego-vehicle IDs, frame IDs, weather labels, object classes, and ground-truth boxes.
- Use an estimated `weather_probability_vector` in the policy state. Do not expose future weather or an unobservable ground-truth weather variable to the runtime policy.
- Calibrate estimators and weather mixtures on training/validation data; reserve test sequences and seeds for final evaluation.

The manuscript reports the following proposed-policy weather-specific reference values:

| Weather | mAP@0.5 | Offload rate | Mean latency |
|---|---:|---:|---:|
| Clear | 84.5% | 77% | 47.3 ms |
| Light rain | 79.2% | 81% | 55.6 ms |
| Heavy rain | 68.4% | 91% | 71.5 ms |
| Dense smog | 62.1% | 95% | 79.2 ms |

These values are reference results, not measured outputs of the current package.

---

## 9. Inference Latency and Hardware Profiling

Run the included profiler:

```bash
python scripts/profile_detector.py \
  --weights runs/bdd100k/yolov8s/weights/best.pt \
  --source "data/bdd100k_yolo/images/benchmark/*.jpg" \
  --device 0 \
  --repeats 300
```

The current profiler performs repeated YOLO predictions using the first image matched by the supplied source pattern. It reports mean, median, P95, and standard deviation for that repeated inference procedure. It is a model-inference timing utility, not a complete dataset-wide benchmark and not an end-to-end vehicle-to-edge latency measurement.

For defensible hardware profiling:

- Warm up the accelerator before recording timings.
- Synchronize the GPU before and after timed inference.
- Fix the image size, batch size, precision, and preprocessing.
- Record clock and power settings where possible.
- Use a representative set of held-out images, not only one image, for a broader benchmark.
- Report mean, SD, P50, and P95, as well as GPU and software metadata.
- Measure power on the actual target device if making vehicle-side or MEC energy claims.

Do not substitute cloud-GPU energy for onboard or MEC energy.

---

## 10. CARLA 0.9.15 Data Collection

Start the CARLA 0.9.15 server, ensure that the matching Python API is available, and run the collector.

Clear weather:

```bash
python -m greencav.carla_collect \
  --frames 1000 \
  --weather clear \
  --out data/carla/clear
```

Rain:

```bash
python -m greencav.carla_collect \
  --frames 1000 \
  --weather rain \
  --out data/carla/rain
```

Fog:

```bash
python -m greencav.carla_collect \
  --frames 1000 \
  --weather fog \
  --out data/carla/fog
```

The collector saves camera images, frame metadata, and projected 2D boxes for supported actor types. Its current supported labels include a subset of the BDD100K taxonomy, such as pedestrians, cars, trucks, buses, motorcycles, bicycles, traffic lights, and traffic signs, subject to the actor types available in the selected CARLA map and scenario.

### 10.1 CARLA annotation validation

Before using these annotations as ground truth:

- Inspect projected boxes visually across different distances, camera angles, and occlusion levels.
- Compare the projection implementation with the official CARLA camera and bounding-box examples.
- Check image dimensions, camera calibration, coordinate transforms, clipping, and actor visibility.
- Record the CARLA version, map, weather, random seed, frame count, and supported categories.
- Document unsupported categories and any annotation exclusions.

The current collector is not a complete replacement for BDD100K's 10-class annotation set. Its three weather presets also do not directly reproduce the four weather proportions in the manuscript.

---

## 11. CARLA–SUMO Co-Simulation

Use the official synchronization script distributed with the CARLA 0.9.15 installation. Depending on the release package, locate the script in the supplied SUMO co-simulation examples and pass its actual path to the project launcher.

A smoke-test command is:

```bash
python scripts/run_official_sumo_bridge.py \
  --bridge-script /path/to/CARLA_0.9.15/Co-Simulation/Sumo/run_synchronization.py \
  --bridge-args "--help"
```

Review the help output for the exact installed release. Then pass the documented SUMO configuration and synchronization options through `--bridge-args`.

A separate CARLA/SUMO smoke test, using an example route supplied with the installation, can be run as follows:

```bash
export SUMO_HOME=/path/to/sumo-1.19.0
cd /path/to/CARLA_0.9.15/Co-Simulation/Sumo
python3 run_synchronization.py examples/Town01.sumocfg --sumo-gui
```

This example assumes that the CARLA server is already running and that SUMO 1.19 binaries are on `PATH`. Use the paths and example configuration included in your actual installation.

The paper does not identify its CARLA town or traffic routes. Town01 is therefore only a smoke-test example; it is not evidence that Town01 generated the manuscript results.

CARLA–SUMO synchronization provides traffic and vehicle coordination, but it does not by itself implement the research-specific V2I communication, MEC offloading, task queues, XAI, or PPO framework.

### 11.1 Required co-simulation configuration

1. Select and document the CARLA town.
2. Generate a compatible OpenDRIVE-to-SUMO road network and verify coordinate, lane, and spawn alignment.
3. Create SUMO route files and record traffic flows, vehicle categories, speed limits, traffic-light settings, and random seeds.
4. Select and document the co-simulation time step. A `0.05 s` step is supported by CARLA's SUMO example, but it is an engineering choice unless confirmed by the manuscript. A 10 Hz sensor event can be scheduled every two such steps.
5. Enable synchronous mode for CARLA, SUMO, sensors, and the scenario runner. Verify timestamp and actor-ID consistency.
6. Define RSU locations, road-segment coverage, uplink estimates, and the vehicles sharing each MEC server. The manuscript does not fully specify RSU count, placement, propagation loss, or mobility/channel coupling.
7. Run at least 100 synchronized steps and fail validation if image timestamps, vehicle identities, or bounding-box transforms drift.
8. Capture visual demonstrations separately from the measurement pipeline. A driving animation alone does not establish energy, latency, or mAP results.

---

## 12. Manuscript Simulation Specification

The following settings are reported in the manuscript and should be treated as the reference configuration.

| Parameter | Manuscript setting | Implementation requirement |
|---|---|---|
| CARLA version | `0.9.15` | Use matching server and client versions. |
| SUMO version | `1.19` | Use SUMO 1.19.x; a newer version is not automatically equivalent. |
| Tasks per run | **30,000** | Generate exactly 30,000 task-release records per run. |
| Paired evaluation runs | **30** | Use 30 independent paired scenario seeds. |
| Sensor/task rate | **10 Hz** | One task every 0.1 s per monitored vehicle. |
| Deadline | **100 ms** | Measure from task release. |
| Clear weather | **87% / 26,100 tasks** | Use ground-truth weather labels for offline evaluation. |
| Light rain | **8% / 2,400 tasks** | Define a reproducible weather preset and camera effect. |
| Heavy rain | **3% / 900 tasks** | Define a reproducible adverse-weather preset. |
| Dense smog | **2% / 600 tasks** | Define and document the fog/smog approximation. |
| Default traffic | **Moderate: 30–50 vehicles/km** | Model density per road length; do not interpret this as 30–50 vehicles total. |
| Radio model | Effective RSU-assisted V2I | Use a synthetic or trace-driven link model, not a claim of physical packet-level C-V2X validation. |
| Carrier/bandwidth | `5.9 GHz / 10 MHz` | Record as link-model metadata. |
| Link rates | `5 / 15 / 30 Mbit/s` | Poor, medium, and good states. |
| Link-state stationary distribution | `5% / 35% / 60%` | Validate the implemented Markov chain. |
| RF transmit power | `20 dBm`, maximum `23 dBm` | Distinguish radiated RF power from modem electrical input power. |
| Receiver noise figure | `9 dB` | Use in the noise calculation. |
| Raw sensor-data envelope | `2.5–5.0 MB` | Do not transmit the raw frame in the split-inference path. |
| Transmitted feature payload | **25,000 bytes** | Measure the serialized feature and document what the byte count includes. |
| Vehicle-side compute profile | Jetson-class, effective 1.8 GHz | Treat as a calibrated compute profile unless measured on target hardware. |
| Edge compute profile | Xeon-class CPU with RTX 3090-class GPU | Treat as a calibrated profile unless measured on the specified hardware. |
| Driving speed | `60 km/h` | Use for scenario-level energy conversion; it need not be the instantaneous CARLA speed. |
| Traction energy | `150 Wh/km` | Manuscript reference conversion. |
| Nominal range | `400 km` | Used for range-loss estimates. |
| Carbon intensity | `0.40 kg CO₂e/kWh` | Scenario assumption, not a field measurement. |

For one ego vehicle generating all 30,000 tasks at 10 Hz, the logical simulation interval is **3,000 seconds (50 minutes)**. If the 30,000 tasks are distributed across multiple vehicles, the total simulation duration will differ. The manuscript does not resolve this distinction; define the task-generation timeline and monitored-vehicle count before running the experiment.

---

## 13. V2I Communication Model

The manuscript uses a three-state effective-link model.

| Link state | Stationary probability | Rate | Pure transfer time for 25,000 bytes | Overhead | Uplink latency | Delivery success | Radio energy per offloaded task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poor | 0.05 | 5 Mbit/s | 40.00 ms | 5.2 ms | 45.20 ms | 97.9% | 0.093 J |
| Medium | 0.35 | 15 Mbit/s | 13.33 ms | 5.2 ms | 18.53 ms | 99.5% | 0.045 J |
| Good | 0.60 | 30 Mbit/s | 6.67 ms | 2.3 ms | 8.97 ms | 99.9% | 0.028 J |

The reference calculations are:

```text
N_total_dBm = -174 + 10 * log10(10,000,000) + 9
            = -95 dBm

T_min_sec   = (25,000 * 8) / (R_Mbit_s * 1,000,000)
T_up_sec    = T_min_sec + T_overhead_sec[state]

P_modem_W   = 10^((20 - 30) / 10) / amplifier_efficiency
              + circuit_power_W[state]

E_radio_J   = P_modem_W * T_up_sec
```

### 13.1 Missing Markov-chain specification

The manuscript provides the stationary distribution:

```text
nu = [0.05, 0.35, 0.60]
```

However, it does not provide the actual 3 × 3 transition matrix `P`.

Either recover the original matrix from the authors or choose and document a new matrix satisfying:

```text
nu @ P = nu
```

Different matrices can share the same stationary distribution while producing different temporal correlations, burstiness, and deadline tails.

Also document:

- The initial link-state distribution.
- The state-transition time unit.
- Any correlation with vehicle position or weather.
- The independent random-number generator stream.
- The link-failure sampling procedure.

Do not generate independent link states and describe them as a temporally correlated Markov process.

### 13.2 Radio-power and failure accounting

- `20 dBm` corresponds to **0.1 W of radiated RF power**. It is not the complete electrical power draw of the modem.
- Include amplifier efficiency and circuit/baseband power when estimating modem input power.
- Use the manuscript's delivery-success values `sigma[state] = 0.979, 0.995, 0.999`.
- Log the random draw and the link state for every attempted offload.
- Record delay and loss per task. A feature must not arrive at the edge earlier than its transmission time.
- Charge failed transmissions for the radio work actually expended. Do not charge MEC inference energy for a task that never reached the MEC server.

The manuscript's reference value of approximately **0.035 J per offloaded task** is obtained by weighting each link state's radio energy by both its state probability and its offloading fraction, then dividing by the total offloading share. It is not obtained by averaging radio energy using only the link-state probabilities.

---

## 14. Discrete-Event Local and MEC Queues

The simulator must represent actual jobs, service times, and resource contention. Adding a fixed average queue delay to every task is not an adequate substitute for a queue model.

### 14.1 Minimum task record

Represent each task using fields such as:

```text
task_id, vehicle_id, released_at, deadline_at, weather, link_state,
selected_detector, execution_location, payload_bytes, arrival_time,
service_begin, service_end, completion_time, drop_reason,
stage_times, energy_components
```

### 14.2 Queue and service requirements

- Each vehicle has a local inference resource and a bounded local buffer.
- The shared MEC server has `g` parallel inference streams and a bounded queue of size `n_max`.
- The manuscript does not specify `g` or `n_max`. Record the selected values and justify them.
- At task release, the policy may receive an **estimated** queue delay, for example `Tq_hat ≈ n_q × mean_service / g`. It must not receive future-realized queue delay.
- Calculate actual queue waiting time from event timestamps: `Tq = service_start - arrival_at_queue`.
- A reference offered-load approximation is `lambda_e = 10 × sum_v(mean_offload_fraction_v)` tasks/s and `rho_e = lambda_e × mean_service / g`. This approximation does not replace discrete-event simulation.
- The manuscript's moderate-traffic reference is approximately 58% edge load and 8.4 ms edge queue delay. Calibrate a realistic distribution; do not assign 8.4 ms to every task.
- If reproducing the local-only reference mean of 112.4 ms, model local backlog and throttling explicitly. Isolated detector inference times alone do not explain that value.
- Record local-buffer overflow, edge-buffer overflow, and link failure as separate drop reasons.
- Account for preprocessing, radio transmission, and partial computation already expended by a dropped task.
- Define whether jobs continue or are cancelled after missing their deadlines.

The manuscript separates task drops from deadline violations:

```text
drop indicator: xi = 1
deadline violation: v = (1 - xi) * I(T > 100 ms)
```

A task that is delivered late is not automatically a drop under this definition.

### 14.3 Traffic-density reference scenarios

| Regime | Density | Edge load | Queue delay | Mean latency | Non-dropped requests per 1,000* | Energy per 1,000 | Tasks/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sparse | 10–20 vehicles/km | 25% | 2.1 ms | 43.1 ms | 998 | 1,189 J | 0.84 |
| Moderate | 30–50 vehicles/km | 58% | 8.4 ms | 49.3 ms | 995 | 1,297 J | 0.77 |
| Dense | 70–100 vehicles/km | 92% | 24.5 ms | 68.4 ms | 976 | 1,479 J | 0.66 |
| Ultra-dense | 120+ vehicles/km | 98% | 56.8 ms | 121.7 ms | 930 | 2,163 J | 0.43 |

\* The manuscript labels this column as successes per 1,000 requests. The moderate value of 995/1,000 corresponds to a 0.5% drop rate, but it does not necessarily represent on-time delivery. Confirm the intended definition. If the drop rate is 0.5% and late-delivered tasks account for 0.6% of all released tasks, the on-time non-dropped rate would be approximately 98.9%, assuming a common denominator.

---

## 15. Uncertainty and Explainable AI

### 15.1 Detection uncertainty

For each predicted box, calculate normalized entropy from a valid K-class probability distribution:

```text
uncertainty = H(p) / log(K)
```

Average the result across detections. If no boxes are predicted, set uncertainty to `1`.

If a detector provides only a class confidence or filtered top-class scores, reconstruct or calibrate a valid class-probability distribution before calculating entropy. Raw logits or a single confidence score must not be treated as a complete probability distribution. Fix and document whether uncertainty is calculated before or after NMS.

### 15.2 Dual-head learned explainer

The manuscript describes a saliency head:

```text
S_hat = ReLU(g_theta(F))
```

and a box-alignment reliability score:

```text
r = saliency_mass_inside_predicted_boxes / total_saliency_mass
```

The manuscript refers to a **dual-head** architecture, but it does not provide the complete layer-by-layer design, initialization, loss weights, teacher/distillation procedure, teacher-mask definition, or trained checkpoint.

The following work is therefore required:

1. Define feature-tensor shapes for each detector and specify the saliency head and the second head or task required to justify the term *dual-head*. Define any adapters for YOLO and Faster R-CNN.
2. Select and justify an explanation teacher or target, such as a declared Grad-CAM++ or perturbation-based target. Label this as a new design choice if it is not recovered from the original implementation.
3. Train the head or heads on training data and select checkpoints using internal validation data only.
4. Upsample saliency maps to image resolution, align predicted boxes, clamp saliency mass to nonnegative values, use a small `eps0` to prevent division by zero, and record reliability `r`.
5. Evaluate confidence-drop fidelity, deletion AUC, and insertion AUC on the same held-out examples. Define the deletion fraction `q`, number of steps, masks, fill baseline, and numerical integration procedure.
6. Compare Grad-CAM++, Score-CAM, Sobol, and the learned explainer using comparable image resolution, batch size, hardware, and precision.
7. Calculate threshold `tau` on validation data as the 75th percentile of predictive uncertainty, then freeze it for testing. A fixed validation threshold does not guarantee exactly 25% invocation under shifted test conditions.
8. If a strict 25% online invocation budget is required, define a causal, predeclared adaptive-quantile procedure and report it separately.
9. For a successful task, use:

```text
x_i = I(u_pred_i >= tau)

if XAI is invoked:
    u_i = (1 - omega) * u_pred_i + omega * (1 - r_i)
otherwise:
    u_i = u_pred_i
    r_i = r_previous
```

10. Define causal feedback for failed or dropped tasks. For example, carry forward the previous valid state and set a `feedback_valid` flag rather than inventing a detection result.
11. Feed `(u_i, r_i)` only into the next eligible decision. For concurrent vehicles, use the most recently completed feedback for that vehicle and respect completion timestamps.
12. Record `omega`, the gate threshold, checkpoint hash, per-call costs, execution location, and power consumption.

### 15.3 XAI reference results and cost accounting

The manuscript reports the following comparison:

| Explainer | Fidelity | Time per call | Memory |
|---|---:|---:|---:|
| Grad-CAM++ | 0.72 | 12.5 ms | 310 MB |
| Score-CAM | 0.76 | 25.4 ms | 420 MB |
| Sobol | 0.81 | 38.2 ms | 650 MB |
| Learned explainer | **0.84** | **15.8 ms** | **95 MB** |

These are manuscript reference values, not guaranteed results from this package.

Use the following accounting rules:

- `T_xai_call = 15.8 ms` is charged only when XAI is invoked.
- Realized amortized overhead is the mean of `x_i × 15.8 ms`; at an exact 25% invocation rate, the arithmetic reference is 3.95 ms per task.
- If XAI runs on the MEC, charge its compute and energy to the MEC. If it runs locally, charge them locally.
- Include XAI latency in the critical path if the result is required before task completion. If feedback is asynchronous, document that as a different architecture and use a consistent latency definition.

---

## 16. Per-Task Latency and Energy Accounting

Use seconds internally and convert to milliseconds only for reporting. Store actual event timestamps and per-stage energy for every task.

### 16.1 Latency model

The manuscript's latency equation is:

```text
T_i = T_pre + T_dec + x_i * T_xai
    + (1 - o_i) * T_local(detector)
    + o_i * (T_uplink + T_edge_queue
             + T_edge(detector) + T_return)
```

The manuscript's Table 11 gives the following task-weighted mean contributions:

| Stage | Mean contribution |
|---|---:|
| Preprocessing and on-vehicle encoding | 3.90 ms |
| Local inference, weighted by 21.9% local tasks | 11.60 ms |
| V2I upload, weighted by 78.1% offloaded tasks | 11.00 ms |
| Edge queue waiting, weighted | 6.60 ms |
| Edge inference, weighted | 11.00 ms |
| XAI, 25% × 15.8 ms | 3.95 ms |
| Policy decision | 0.50 ms |
| Result return | 0.70 ms |
| **Total** | **49.25 ms ≈ 49.3 ms** |

These are weighted contributions across all tasks. They are **not** delays to add to every individual task. A local task must not also be charged an edge uplink delay. Calculate each task's applicable stages first, then average each stage over the full run.

**Unresolved numerical difference:** A simple calculation using the link-state shares, offload fractions, and uplink latencies in the manuscript's Table 8 gives approximately 9.87 ms of uplink contribution per released task, whereas Table 11 reports 11.00 ms. Additional timing components, correlations, or unrounded measurements may explain the difference, but the manuscript does not provide a reconciliation. Verify the difference using raw task logs; do not insert an unexplained adjustment.

### 16.2 Energy model

The manuscript's energy accounting is:

```text
E_vehicle = E_pre
          + (1 - o) * (E_local_detector + x * E_xai_local)
          + o * E_radio
          + P_static * T_total

E_system  = E_vehicle
          + o * (E_edge_detector + x * E_xai_edge)
```

Apply the following rules:

- Use one clearly defined accounting boundary.
- Do not count a device's idle/static power both inside a detector profile and again in `P_static × T_total`.
- Attribute dropped-task energy according to the work actually completed.
- Do not attribute MEC energy to a task that failed before reaching the MEC.
- Treat Table 2 local and edge energy values as computation-only reference values, not automatically as full-server or battery energy.
- The manuscript does not provide enough raw numeric values for `k_c`, edge power, amplifier efficiency, circuit power, `P_static`, `E_pre`, and XAI energy to regenerate every system-energy total uniquely. Obtain calibration data or clearly document new assumptions.
- If all workload energy is included, calculate:

```text
energy_kJ = sum(E_system_J) / 1,000
energy_per_task_J = sum(E_system_J) / 30,000
```

State explicitly how dropped tasks are handled and use the same rule in every comparison.

---

## 17. Preference-Conditioned PPO Agent

### 17.1 State vector

For task `i`, the manuscript defines a state containing:

```text
state_i = [
    weather_probability_vector_i,  # Four estimated probabilities
    uncertainty_feedback_prev,    # Previous completed task
    reliability_feedback_prev,    # Previous completed task
    estimated_uplink_rate_i,
    estimated_edge_queue_wait_i,
    estimated_edge_utilization_i,
    local_backlog_i,
    alpha,
    beta
]
```

Use estimated observations, not future-realized queue or network values. Document feature normalization and every estimator. Action masking for an unavailable MEC server is an additional implementation choice and must be described.

### 17.2 Action space

The joint action selects:

- Execution location: `o ∈ {local, edge}`.
- Detector: `m ∈ {YOLOv5s, YOLOv8s, YOLOv8m, Faster R-CNN}`.

This produces **eight joint discrete actions**. Implement either one categorical distribution over all eight actions or two appropriately linked action heads. Record the joint action probability. Document any feasibility constraints.

### 17.3 Cost and reward

The manuscript defines:

```text
q_i = (1 - drop_i)
      * accuracy_for_selected_detector_weather / 100

c_i = alpha * (E_system_i / E_ref)
    + beta  * (T_i / 0.100)
    + psi * (1 - q_i)
    + lambda_deadline * late_i
    + lambda_drop * drop_i

reward = -c_i
beta = 1 - alpha
```

The manuscript does not specify numerical values for `psi`, `lambda_deadline`, `lambda_drop`, or `E_ref`. It also does not fully specify `gamma`, GAE lambda, PPO clipping epsilon, policy/value network architecture, optimizer learning rate, minibatch size, rollout length, update epochs, training iterations, or the preference-sampling distribution. These values must be recovered or explicitly declared as new implementation choices.

If the reward uses an accuracy term based on mAP, remember that mAP is a dataset-level metric, not an exact per-task differentiable detection accuracy. The manuscript uses a profiled, weather-conditioned accuracy surrogate in the reward. Keep that surrogate distinct from genuine offline AP/mAP computed from predictions and ground-truth boxes.

### 17.4 Recommended training sequence

1. Finalize and freeze datasets, splits, and detector-profile caches.
2. Train and freeze detector checkpoints and calibrated split encoders.
3. Train and freeze the XAI component; calibrate `tau` using validation data.
4. Build deterministic, resettable CARLA–SUMO, link, and event-queue environments.
5. Implement and validate all five baselines.
6. Train the fixed-detector energy-aware DRL baseline using its declared restricted state, without the proposed weather, uncertainty, and XAI inputs. Use comparable training budgets and identical test traces.
7. Train the proposed preference-conditioned PPO agent. The manuscript does not specify the preference distribution; values such as `alpha = 0.2, 0.5, 0.8` may be used as documented test preferences, but must not be presented as recovered historical settings.
8. Save model checkpoints, optimizer state, normalization parameters, thresholds, configuration files, and random seeds.
9. Select checkpoints using validation data only. During final evaluation, use deterministic argmax actions rather than sampling from the policy.
10. Evaluate on independent test scenarios. Give all policies the same external random traces where possible.

### 17.5 Data-leakage and reproducibility checks

- [ ] The policy cannot observe future queue, service, weather, or link states.
- [ ] XAI feedback becomes available only after the corresponding inference has completed.
- [ ] Separate random-number generators are used for traffic, weather, network, queues/background load, detector training, and policy training.
- [ ] The seed map is saved for every run.
- [ ] Training, validation, and test frames/routes are isolated.
- [ ] Detector weights and preprocessing remain frozen during policy comparisons unless a different protocol is explicitly declared.
- [ ] Reward normalization and running state statistics are frozen during evaluation.
- [ ] Each run records 30,000 task records with unique IDs and clearly defined metric denominators.

---

## 18. Baselines and Ablation Studies

### 18.1 Six main policies

The manuscript's main comparison contains six policies: five baselines and the proposed method.

1. **Local-only:** Execute every task on the vehicle. Never offload. Specify the detector-selection rule.
2. **Edge-only:** Send every task to the MEC. Specify the detector-selection rule and count link failures and edge-queue drops.
3. **Greedy:** Select the candidate action with the lowest predicted end-to-end latency using only currently available state estimates.
4. **Deadline-aware:** Among actions predicted to meet the 100 ms deadline, select the one with the lowest predicted system energy. Define a fallback for cases in which no action is predicted to meet the deadline.
5. **Energy-aware DRL:** Use PPO with a fixed mid-size detector and restricted state inputs. The manuscript does not identify whether the fixed detector is YOLOv8s or YOLOv8m; resolve and document this choice.
6. **Proposed policy:** Use preference-conditioned PPO to select execution location and detector, using weather estimates, previous XAI feedback, link estimates, and queue information.

### 18.2 Ablation reference targets

| Variant | Latency | Energy | Drops | Deadline violations | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| No XAI feedback/gate | 56.4 ms | 42.1 kJ | 1.8% | 1.9% | 76.5% |
| No energy objective | 48.1 ms | 55.6 kJ | 0.8% | 0.5% | 83.0% |
| No dynamic perception (fixed model) | 52.8 ms | 44.5 kJ | 2.5% | 2.8% | 71.4% |
| Full proposed method | **49.3 ms** | **38.9 kJ** | **0.5%** | **0.6%** | **83.1%** |

These are manuscript reference values.

For scientifically meaningful ablations, state whether each variant is retrained from scratch after a component is removed or evaluated with an input removed from an existing checkpoint. Retraining is generally preferable for architecture or reward ablations. Removing the energy objective changes the reward and therefore requires a clearly documented training protocol.

---

## 19. Experimental Plan: Tables 2–12

The manuscript describes 11 experimental tables. The following plan identifies the work needed to regenerate them from recorded data.

### Experiment A — Detector profiling (Table 2)

- Profile all four detectors on the same documented images.
- Measure local and MEC computation latency and energy separately.
- Record actual AP/mAP, timing distributions, hardware details, and checkpoint hashes.

### Experiment B — Six-policy comparison (Table 3)

- Run 30 independent seeds for each of the six policies.
- Generate 30,000 tasks per policy and seed.
- This gives `30 × 6 × 30,000 = 5,400,000` policy-task evaluations for the main comparison.
- Use the balanced reference preference `alpha = 0.5`, `beta = 0.5`.
- Use the same external scenario seeds across policies. Policy-dependent queue outcomes may differ and should be treated as part of the policy effect.
- Export per-task records and 30 run-level metric rows per policy.
- Report means and sample standard deviations across runs.

### Experiment C — Weather conditions (Table 4)

Evaluate clear, light rain, heavy rain, and dense smog. If enforcing the manuscript's proportions, use 26,100, 2,400, 900, and 600 tasks per 30,000-task run, respectively.

Report per-weather ground-truth mAP, mean entropy, local/edge ratio, latency, deadline violations, and energy. Keep sample counts and class populations visible.

### Experiment D — XAI evaluation (Table 5)

- Evaluate all explainers on the same samples.
- Use consistent mask fractions, image-fill rules, and confidence definitions.
- Measure fidelity, deletion AUC, insertion AUC, execution time, and peak memory.
- Report invocation rate and confidence intervals.

### Experiment E — Preference modes (Table 6)

| Mode | Alpha (energy) | Beta (latency) | Reference energy | Mean latency | Deadline violations | mAP |
|---|---:|---:|---:|---:|---:|---:|
| Energy-saving | 0.8 | 0.2 | 32.4 kJ | 85.2 ms | 18.6% | 82.1% |
| Balanced | 0.5 | 0.5 | 38.9 kJ | 49.3 ms | 0.6% | 83.1% |
| Latency-priority | 0.2 | 0.8 | 51.2 kJ | 38.5 ms | 0.2% | 83.3% |

Use one trained preference-conditioned agent with different preference inputs unless the experiment explicitly compares separately trained agents. The energy-saving reference has an 18.6% deadline-violation rate. A mean latency below 100 ms does not, by itself, establish real-time compliance.

### Experiment F — Ablations (Table 7)

Evaluate the three ablations and the full method on paired test scenarios. Document whether each ablation is retrained. Keep data, training budgets, rewards, and seeds comparable.

### Experiment G — Link sensitivity (Table 8)

| Link rate | Offload-share reference | End-to-end latency reference | Link-success reference |
|---|---:|---:|---:|
| 5 Mbit/s | 32.0% | 92.0 ms | 97.9% |
| 15 Mbit/s | 68.0% | 59.0 ms | 99.5% |
| 30 Mbit/s | 87.8% | 40.0 ms | 99.9% |

State whether each experiment forces a single link state or uses a mixture of link states. Separate conditional results from state-weighted aggregate results.

### Experiment H — Traffic density and scalability (Table 9)

Evaluate the four density regimes in Section 14.3. Define road length, vehicle density, server placement, and background workload. Measure realized server utilization, queue waiting, timeouts, late deliveries, drops, useful throughput, and energy. Do not prescribe the target queue delay as an input.

The ultra-dense reference mean latency is 121.7 ms, which exceeds the 100 ms deadline and illustrates a limitation of the reference system.

### Experiment I — Vehicle energy and carbon (Table 10)

Use the manuscript's reference conversion inputs: 10 Hz task rate, 60 km/h driving speed, 150 Wh/km traction energy, 400 km nominal range, and 0.40 kg CO₂e/kWh carbon intensity. Clearly label this as a scenario-level conversion model, not a field battery or emissions measurement.

### Experiment J — Latency breakdown (Table 11)

Derive the stage contributions from logged per-task timestamps. Local and edge execution branches must be mutually exclusive. Charge XAI only when invoked.

### Experiment K — Statistical analysis (Table 12)

- Save the same metrics for every policy and every seed.
- Compare paired run-level observations, such as Proposed minus Energy-aware DRL and Proposed minus Deadline-aware.
- Use a two-sided Wilcoxon signed-rank test on nonzero paired differences. Record the zero/tie handling and software version.
- Calculate a 95% paired-difference confidence interval, for example by bootstrapping the 30 seed pairs.
- Report the mean paired difference, standard deviation of differences, confidence interval, p-value, effect size, and any multiple-comparison correction such as Holm's procedure.
- Do not describe an independent-group confidence interval as a paired confidence interval.
- The pooled-SD Cohen's `d` is not the same as paired effect size `d_z`.
- A p-value below 0.001 cannot be reconstructed from manuscript means and standard deviations alone; actual paired per-run data are required.

---

## 20. Required Logs and Metric Definitions

### 20.1 Minimum per-task record

A suggested schema for `output/raw_tasks/<policy>/<seed>.parquet` is:

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
uncertainty_fused, feedback_ready_at_s, drop_reason
```

Unknown values, such as completion time after a drop, should be null/NaN and accompanied by an explicit `drop_reason`. Do not replace unknown values with zeros and then include them in average-latency calculations.

The manuscript does not clearly state whether dropped tasks are excluded from mean latency or assigned a timeout latency. Define the convention before evaluation and, if the ambiguity remains, report both completed-only and drop-penalized latency.

### 20.2 Minimum run-level metrics

A suggested schema for `output/runs/metrics.csv` is:

```text
seed, policy, n_released, n_completed, n_dropped,
mean_latency_ms, p95_latency_ms, deadline_violation_pct,
drop_pct, timely_delivered_pct, offloaded_pct,
system_energy_kj, energy_j_per_released_task,
vehicle_energy_j_per_task, xai_invoked_pct,
mAP50, mean_xai_overhead_ms,
mean_edge_wait_ms, avg_server_utilization
```

### 20.3 Metric definitions

Use and publish explicit definitions:

- `drop_pct = 100 × sum(dropped) / n_released`
- `deadline_violation_pct = 100 × sum((not dropped) AND (latency_ms > 100)) / n_released`, following the manuscript's indicator unless another denominator is explicitly labelled.
- `timely_delivered_pct = 100 × sum((not dropped) AND (latency_ms <= 100)) / n_released`
- `offloaded_pct = 100 × sum(offload == 1) / n_released`; this measures attempted offloads. If successful offloads are also reported, label them separately.
- `energy_per_task = total workload energy, including failed attempts, divided by n_released`, subject to the declared accounting boundary.
- Define which tasks are included in mean and P95 latency. If drop handling remains ambiguous, report completed-only and drop-penalized variants.
- Calculate mAP@0.5 from actual predicted boxes and ground-truth labels on held-out frames. Specify how dropped and late detections are handled. A missing prediction may be treated as an empty prediction or excluded under a separately labelled protocol.
- Distinguish non-dropped delivery from on-time, correct, usable delivery. Do not silently change the definition of success or goodput.
- Calculate run-level means and standard deviations across the **30 independent runs**. Do not treat all pooled task records as independent experimental replicates.

---

## 21. Reference Calculations

The following checks are algebraic consistency checks using manuscript-reported values. They are **not experimental results** and must not be used to synthesize task records that are later presented as measured data.

```python
# Standard-library-only checks of manuscript summary values.

shares = [0.87, 0.08, 0.03, 0.02]
lat = [47.3, 55.6, 71.5, 79.2]
eng = [1.25, 1.48, 1.75, 1.92]
off = [77.0, 81.0, 91.0, 95.0]
map_weather = [84.5, 79.2, 68.4, 62.1]

weighted = lambda values: sum(p * x for p, x in zip(shares, values))

assert round(weighted(lat), 1) == 49.3
assert abs(weighted(eng) * 30_000 / 1_000 - 38.904) < 1e-6
assert round(weighted(off), 1) == 78.1
assert round(weighted(map_weather), 1) == 83.1  # Summary approximation only.
assert abs(0.25 * 15.8 - 3.95) < 1e-10
assert abs(sum([3.9, 11.6, 11.0, 6.6, 11.0, 3.95, 0.5, 0.7]) - 49.25) < 1e-10

# 20 dBm corresponds to 0.1 W of radiated RF power.
assert abs(10 ** ((20 - 30) / 10) - 0.1) < 1e-10

link_share = [0.05, 0.35, 0.60]
link_offload = [0.32, 0.68, 0.878]
link_uplink_ms = [45.2, 18.5, 9.0]
link_radio_j = [0.093, 0.045, 0.028]

overall_offload = sum(
    p * o for p, o in zip(link_share, link_offload)
)

radio_energy_per_offloaded_task = sum(
    p * o * e
    for p, o, e in zip(link_share, link_offload, link_radio_j)
) / overall_offload

uplink_per_released_task_ms = sum(
    p * o * t
    for p, o, t in zip(link_share, link_offload, link_uplink_ms)
)

print("Weather-weighted latency (ms):", weighted(lat))
print("Weather-weighted energy (kJ/30,000 tasks):",
      weighted(eng) * 30_000 / 1_000)
print("Link-weighted offload (%):", overall_offload * 100)
print("Radio energy per offloaded task (J):",
      radio_energy_per_offloaded_task)  # Approximately 0.035 J.
print("Uplink contribution per released task (ms):",
      uplink_per_released_task_ms)  # Approximately 9.87 ms.
print("Derived CO2e (g/km):", (1.2968 * 10 / 60) * 0.4)
```

The approximately 9.87 ms uplink contribution should be investigated against the 11.00 ms contribution in Table 11. Do not add an unexplained adjustment to force the values to agree.

A synthetic dataset fitted to published means may be useful for software or plotting tests, but it is not independent empirical evidence and must never be presented as measured experimental data.

---

## 22. Known Gaps and Reproduction Risks

The following items must be resolved before claiming a faithful implementation or independent reproduction.

| ID | Missing detail or potential inconsistency | Required action |
|---|---|---|
| B01 | No complete original source code or checkpoints were identified in the paper-associated repository when the source guide was prepared. | Obtain the original implementation or explicitly describe this work as a new reimplementation. |
| B02 | The CARLA town, actor-density generator, routes, camera setup, simulator step, and weather transitions are not fully specified. | Version and hash all scenario configurations and report every selected value. |
| B03 | Dataset, class taxonomy, ground-truth procedure, weights, and train/test splits are not fully documented. | Create and audit labelled frames and held-out evaluation splits. |
| B04 | The full detector-by-weather accuracy matrix `A(m,w)` is absent. | Measure the matrix; do not infer arbitrary values from aggregate tables. |
| B05 | The 25 kB latent representation lacks encoder, split-layer, quantization, decoder, and accuracy details. | Implement and measure the actual 25,000-byte split-inference path. |
| B06 | The Markov stationary distribution is given, but the transition matrix is not. | Recover the matrix or disclose a newly calibrated matrix and test its temporal behaviour. |
| B07 | MEC stream count, buffer size, scheduling, overflow, and load parameters are incomplete. | Record queue settings and calibrate them using unbiased traces. |
| B08 | Energy constants and static-power calibration are incomplete. | Provide measured power data or label model inputs as calibrated assumptions. |
| B09 | PPO architecture, hyperparameters, reward penalties, and training budgets are incomplete. | Recover or fully specify these values and conduct sensitivity analyses. |
| B10 | XAI teacher, dual-head architecture, losses, `omega`, masks, and threshold protocol are incomplete. | Implement and document the complete XAI pipeline, including causal feedback. |
| B11 | mAP is used as a weather-weighted scalar in the reward and summary table. | Distinguish the reward surrogate from genuine dataset-level AP/mAP. |
| B12 | Table 8 implies approximately 9.87 ms weighted upload time, while Table 11 reports 11.00 ms. | Reconcile the difference using raw stage logs; do not insert an unexplained adjustment. |
| B13 | Table 9's reported successes appear consistent with non-drop counts, not necessarily on-time delivery. | Report drop, late-delivery, and on-time success rates separately. |
| B14 | Table 12's confidence intervals are described as independent-group approximations despite paired tests. | Calculate valid paired confidence intervals from seed-level data. |
| B15 | Dropped-task latency, energy, and missing-detection treatment in mAP are unclear. | Define denominators, timeout conventions, and missing-prediction handling. |
| B16 | The relationship between 30,000 tasks at 10 Hz and a multi-vehicle workload duration is unclear. | Specify the monitored vehicles, task generator, and concurrent traffic. |
| B17 | Encoder and XAI placement in the local/edge execution path is not fully specified. | Measure the split path and publish a complete stage timeline. |
| B18 | Detector compute and energy profiles are conceptual or calibrated rather than verified on the stated hardware. | Do not claim hardware validation without instrumented target-device measurements. |

### Acceptance standard

A system may be described as **implemented** when its components are present and functionally tested. It may be described as a **faithful implementation** only after the relevant specification gaps above have been addressed and the required artifacts are available. It may be described as an **independent reproduction** only after raw test runs have been completed and the results have been analysed with valid uncertainty estimates. Exact agreement with the manuscript is not required, but all differences must be disclosed.

---

## 23. Staged Execution Workflow

### 23.1 Commands implemented in the current package

The following commands correspond to scripts that are included in this repository:

```bash
# Environment and provenance
python -m greencav.provenance

# BDD100K preprocessing
python scripts/prepare_bdd100k.py

# YOLOv8 training
python scripts/train_yolo.py --model yolov8s --epochs 100 --batch 16 --device 0
python scripts/train_yolo.py --model yolov8m --epochs 100 --batch 8 --device 0

# YOLOv8 evaluation
python scripts/evaluate_yolo.py --weights runs/bdd100k/yolov8s/weights/best.pt
python scripts/evaluate_yolo.py --weights runs/bdd100k/yolov8m/weights/best.pt

# YOLOv5s training
bash scripts/train_yolov5s.sh

# Faster R-CNN training and evaluation
python scripts/train_fasterrcnn.py --epochs 20 --batch 4 --device cuda
python scripts/evaluate_fasterrcnn.py --checkpoint checkpoints/fasterrcnn_bdd100k.pth

# YOLO inference profiling
python scripts/profile_detector.py \
  --weights runs/bdd100k/yolov8s/weights/best.pt \
  --source "data/bdd100k_yolo/images/benchmark/*.jpg" \
  --device 0 --repeats 300

# BDD100K weather manifests
python scripts/export_weather.py \
  --labels data/bdd100k/labels/bdd100k_labels_images_val.json \
  --images data/bdd100k/images/100k/val

# CARLA collection (CARLA server must already be running)
python -m greencav.carla_collect --frames 1000 --weather clear --out data/carla/clear

# Official CARLA–SUMO bridge launcher
python scripts/run_official_sumo_bridge.py \
  --bridge-script /path/to/run_synchronization.py \
  --bridge-args "--help"

# Empirical profiles and result aggregation
python scripts/build_empirical_profiles.py
python scripts/merge_results.py
```

The package also includes:

```bash
bash scripts/run_all_detectors.sh
bash scripts/run_all_evaluations.sh
```

Review these scripts and adjust batch sizes, device settings, and file paths to match the available hardware and completed training runs.

### 23.2 Proposed workflow for components not yet implemented

The following commands describe a recommended future interface for the full research system. They are **placeholders** and will not run until the corresponding modules and scripts have been developed.

```bash
# Phase 1: Validate the simulator and export scenarios
bash scripts/01_smoke_test.sh
bash scripts/02_generate_scenarios.sh
python -m src.cosim.sensor_export --config config/simulation.yaml --split train
python -m src.cosim.sensor_export --config config/simulation.yaml --split val
python -m src.cosim.sensor_export --config config/simulation.yaml --split test

# Phase 2: Profile detectors and split-inference paths
bash scripts/03_profile_detectors.sh

# Phase 3: Train and validate the XAI component
bash scripts/04_train_xai.sh

# Phase 4: Train the proposed PPO agent, energy-aware DRL baseline, and ablations
bash scripts/05_train_ppo.sh

# Phase 5: Run the independent 30-seed paired evaluation
bash scripts/06_evaluate_paired.sh

# Phase 6: Aggregate tables, statistical tests, and figures
bash scripts/07_analyze_results.sh
```

### 23.3 Recommended pilot sequence

Do not begin with the full multi-million-task experiment. Validate the system in stages:

1. Run 100 tasks with one vehicle, one weather condition, and forced local execution. Validate timestamps and energy accounting.
2. Run 100 forced-offload tasks at each of the three link rates. Validate payload size, transfer delay, and drop handling.
3. Run 1,000 tasks with two vehicles sharing one MEC server. Validate queue contention and causal feedback.
4. Run 1,000 tasks using all four detectors and varied weather. Validate bounding boxes, mAP, and XAI invocation rate.
5. Run a small PPO training experiment with three preference settings. Confirm that reward behaviour is sensible and the policy improves beyond a random baseline.
6. Run one complete 30,000-task experiment for all six policies and all four weather categories. Inspect distributions and failure cases, not only averages.
7. Freeze configurations and checkpoints, then run the 30 paired test seeds.
8. Run ablations and sensitivity experiments without using test results to tune training choices. Generate all tables and figures from the recorded outputs.

---

## 24. Troubleshooting and Diagnostic Checks

### 24.1 If latency frequently exceeds 100 ms

- Confirm that the transmitted payload is 25 kB, not the original 2.5–5 MB image.
- Check units carefully: Mbit/s is not MB/s, and seconds are not milliseconds.
- Confirm that XAI is invoked only when the gate is triggered; do not charge every task the full 15.8 ms unless every task actually invokes XAI.
- Check encoder and split-inference costs and synchronize GPU timing correctly.
- Ensure that host simulation frame time is not being reported as detector inference time.
- Inspect MEC backlog and concurrent vehicle load.
- Check detector selection. The manuscript's Faster R-CNN local inference reference is 145 ms.
- Inspect P95 and P99 latency; a low mean can conceal deadline violations and drops.

### 24.2 If energy is unexpectedly high or low

- Confirm that 20 dBm converts to 0.1 W of radiated RF power, while total modem electrical consumption also includes amplifier and circuit power.
- Ensure that preprocessing, static power, local/edge inference, radio, and XAI energy are each counted exactly once.
- Charge failed transmissions for the radio work already performed.
- Do not charge MEC inference energy to tasks that never reached the MEC server.
- Confirm that system energy includes the declared vehicle, radio, and MEC components.
- If calibrated profiles are used, document calibration residuals and compare each stage with the manuscript's reference profiles.

### 24.3 If mAP is unexpectedly low or high

- Check the dataset, class mapping, ground-truth export, and CARLA occlusion handling.
- Use consistent detector checkpoints, image preprocessing, and thresholds.
- Verify that full and split inference produce comparable predictions if equivalence is claimed.
- Recalculate AP from actual predicted and ground-truth boxes.
- Do not assign the manuscript's 83.1% mAP to individual tasks as though it were a measured per-task accuracy.
- Verify weather-specific results on held-out images.
- Document how dropped and late frames contribute to evaluation.

### 24.4 If results appear implausibly good

- Check that the policy cannot access future link, queue, or weather information.
- Check that test data were not used for training or XAI threshold calibration.
- Verify independent test seeds and paired external traces.
- Ensure that target values are not directly assigned as simulator outputs or hard-coded policy behaviour.

---

## 25. Reproducibility Deliverables

Before submitting a reproducible research package, include:

- [ ] Complete source code, dependency lock file, setup instructions, licence information, and third-party attributions.
- [ ] Dataset access instructions or export scripts, class mapping, train/validation/test manifests, and checksums.
- [ ] CARLA town, SUMO network and routes, four weather presets, RSU coordinates, sensor configuration, and 30 external seeds.
- [ ] Four detector checkpoints or verifiable third-party hashes.
- [ ] Split-inference encoder/decoder and XAI checkpoints.
- [ ] PPO model and normalization checkpoints, hyperparameters, reward weights, and training curves.
- [ ] Per-task traces for every policy and seed, including detector predictions, ground truth, profiling data, and failure reasons.
- [ ] Run-level metrics with 30 runs per comparison, with ablations and sensitivity experiments clearly separated.
- [ ] Tables 2–12 regenerated from recorded data rather than copied into plotting scripts.
- [ ] Paired Wilcoxon analysis, paired confidence intervals, effect sizes, multiple-testing policy, statistical code, and environment versions.
- [ ] Figure PDFs and, optionally, a rendered demonstration video clearly identified as an illustration rather than evidence of measured system performance.
- [ ] A `deviations_from_manuscript.md` file documenting differences in settings, assumptions, and results.
- [ ] A repeatable pipeline, such as `make reproduce`, that checks prerequisites and reports missing artifacts clearly.

---

## 26. Completion Checklist

### A. Environment and simulation

- [ ] CARLA 0.9.15 and SUMO 1.19.x versions verified.
- [ ] Map, routes, simulation time step, sensors, RSUs, vehicles, and weather configurations committed.
- [ ] CARLA–SUMO timestamp and actor synchronization validated.
- [ ] Thirty paired external test seeds and dataset split isolation recorded.

### B. Perception and communication

- [ ] All four detectors produce actual bounding-box predictions.
- [ ] AP, latency, and power measurements have been audited.
- [ ] The encoder produces a validated 25,000-byte payload.
- [ ] The edge decoder accepts the payload and completes the intended inference path.
- [ ] The Markov link satisfies its stationary-distribution and transfer/failure tests.
- [ ] Local/MEC queues and the energy ledger pass timestamp and accounting checks.

### C. Learning and explainability

- [ ] XAI training and fidelity protocols are finalized.
- [ ] The uncertainty gate is calibrated on validation data and frozen for testing.
- [ ] The proposed eight-action PPO agent is trained with the declared preference inputs.
- [ ] Five baselines and three ablations are implemented with fair comparison budgets.

### D. Experimental results

- [ ] Raw per-task and per-run metrics are available and independently aggregable.
- [ ] Definitions for mean/P95 latency, violations, drops, offloading, energy, and mAP are fixed.
- [ ] All tables and figures are regenerated from experimental logs.
- [ ] Wilcoxon tests and confidence intervals use the actual 30 paired run differences.
- [ ] Differences from the manuscript's reference values—49.3 ms, 38.9 kJ, and 83.1% mAP—are documented transparently.

---

## 27. Recommended Implementation Priority

For an efficient and scientifically defensible implementation, follow this order:

1. Install CARLA 0.9.15 and SUMO 1.19.x. Validate synchronization and record camera frames, moving vehicles, and ground-truth annotations.
2. Train and evaluate all four detectors on a documented dataset. Obtain actual bounding-box mAP and local/edge compute profiles.
3. Implement the 25,000-byte feature encoder/decoder and three-state link model. Verify payload size and upload time.
4. Implement multi-vehicle local and MEC queues, event timestamps, deadline handling, and complete energy accounting.
5. Train the dual-head XAI component, calibrate `tau` on validation data, and implement causal feedback to the next eligible decision.
6. Implement the reward and joint eight-action PPO policy. Train a separate, fair energy-aware DRL baseline.
7. Run the six main policies, three ablations, and weather, preference, link, and traffic-density experiments.
8. Use the 30 paired test seeds and 30,000 tasks per run to generate tables, figures, measured mAP, Wilcoxon tests, and confidence intervals.
9. Compare the measured results with the manuscript's reported targets. Archive all assumptions, deviations, source code, datasets or access instructions, configurations, and checkpoints.

**Bottom line:** Verifying the manuscript's reported metrics requires working CARLA–SUMO scenarios, actual detector inference, a validated 25 kB split-inference path, a stochastic V2I/MEC simulator, trained XAI and PPO components, and 30 paired raw evaluation runs. Aggregate manuscript tables alone are insufficient to reconstruct the original experiment exactly.

---

## 28. References and Setup Documentation

1. [CARLA installation guide](https://carla.readthedocs.io/en/latest/start_quickstart/)
2. [CARLA–SUMO co-simulation guide](https://carla.readthedocs.io/en/latest/adv_sumo/)
3. [SUMO installation documentation](https://sumo.dlr.de/docs/Installing/)
4. [SUMO TraCI documentation](https://sumo.dlr.de/docs/TraCI/)
5. [BDD100K dataset](https://www.vis.xyz/bdd100k/)
6. [BDD100K documentation](https://doc.bdd100k.com/download.html)
7. [Paper-associated GitHub repository and walkthrough reference](https://github.com/ImranKhalid-IT/Towards-Green-Connected-Autonomous-Vehicles-Explainable-Perception-Aware-Multi-Objective-Off)

---

## Reproduction Integrity Notice

Matching the manuscript's aggregate numbers by adjusting a simulator is not, on its own, evidence of independent reproduction. Report the actual data, code, assumptions, task-level samples, random seeds, hardware measurements, and statistical variation. Clearly distinguish manuscript-reported reference values from newly measured results.
