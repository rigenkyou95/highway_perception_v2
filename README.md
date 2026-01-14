# Highway Perception V2

**Highway Perception V2** is a **research-oriented monocular perception system** for highway scenarios, integrating **lane detection, drivable area segmentation, vehicle detection, and monocular depth estimation** with explicit **camera geometry and calibration**.

This repository serves as the **official experimental codebase** for research on **geometry-aware monocular road perception**.

---

## Overview

- **Input**: Single monocular RGB image or video (fixed camera)
- **Outputs**:
  - Lane lines
  - Drivable area
  - Vehicles (bounding boxes)
  - Metric-aligned depth map (geometry-guided)

The system is designed for:
- Highway surveillance
- Driver assistance research
- Geometry-constrained perception under perspective distortion

---

## Method Overview

### System Pipeline

```text
Monocular Image
      │
      ▼
┌──────────────────────────┐
│ Lane & Drivable Area Seg │  ← TwinLiteNet++
└───────────┬──────────────┘
            │
            │ Lane geometry / road structure
            ▼
┌──────────────────────────┐
│ Camera Geometry Module   │
│  • Intrinsics            │
│  • Camera Height         │
│  • Pitch / Roll / Yaw    │
│  • IPM / BEV Projection  │
└───────────┬──────────────┘
            │
            │ Sparse metric depth anchors
            ▼
┌──────────────────────────┐
│ Monocular Depth Network  │  ← Depth Anything V2 (Tiny)
│  (relative depth)        │
└───────────┬──────────────┘
            │
            │ Geometry-guided scale alignment
            ▼
┌──────────────────────────┐
│ Metric Depth Estimation  │
└──────────────────────────┘
````

### Parallel Branch

```text
Monocular Image
      │
      ▼
┌──────────────────────────┐
│ Vehicle Detection        │  ← YOLOv11n
└──────────────────────────┘
```

---

## Key Design Principles

* **Geometry before learning**
  Lane structure and camera calibration provide reliable metric constraints.

* **Lightweight models**
  All networks are selected for real-time feasibility in highway scenarios.

* **Explicit calibration**
  Camera intrinsics and extrinsics are treated as first-class components, not hidden assumptions.

* **Modular research code**
  Each component can be replaced, evaluated, or extended independently.

---

## Repository Structure

```text
highway_perception_v2/
├── calib/                  # Camera calibration files
│   └── cam_v002/            # Versioned camera configuration
├── calib_videos/            # Calibration videos (not tracked)
├── configs/                 # YAML-based configuration files
│   └── camera/
├── detector/                # Vehicle detection wrapper
├── models/
│   ├── seg/                 # Lane & drivable area segmentation
│   └── depth/               # Monocular depth estimation
├── scripts/
│   ├── infer/               # Inference & evaluation scripts
│   └── train/               # Training utilities
├── third_party/             # External repositories (Git submodules)
├── outputs/                 # Inference results (ignored)
├── notes/                   # Experiment logs & research notes
├── README.md
└── .gitignore
```

> `outputs/`, `runs/`, and checkpoints are intentionally excluded from version control.

---

## Quick Usage

### Lane & Drivable Area Segmentation

```bash
python scripts/infer/test_twinlitenet_pp.py
```

### Vehicle Detection

```bash
python scripts/infer/test_yolo11n.py
```

### Monocular Depth Estimation

```bash
python scripts/infer/test_depth_anything_v2.py
```

### Full Pipeline (Single Image)

```bash
python scripts/infer/test_full_system_single_image.py
```

---

## Camera Calibration

Calibration data are stored in `calib/` and include:

* Camera intrinsics
* Camera height
* Pitch / roll / yaw
* Calibration quality and data source metadata

Supported operations:

* Lane-based pitch estimation
* Roll / yaw estimation from vanishing geometry
* IPM-based BEV projection

---

## Research Scope

This project focuses on:

* Geometry-aware monocular depth estimation
* Lane-based metric scale recovery
* Highway-specific perception constraints
* Robust perception under occlusion and perspective distortion

This codebase is intended **for academic research purposes only** and is **not a production-ready ADAS system**.

---

## Dependencies & Submodules

This repository includes the following projects as **Git submodules**, each pinned to a specific commit for reproducibility:

* **TwinLiteNet / TwinLiteNet++** – lane & drivable area segmentation
* **Depth Anything V2** – monocular depth estimation
* **Ultralytics YOLO** – vehicle detection

---

## Author

**Rigenkyou**
PhD Researcher
Monocular Road Perception · Geometry-aware Vision · Highway Surveillance


