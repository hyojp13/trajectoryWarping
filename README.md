# trajwarp — Kinematic Non-Rigid Spatio-Temporal Trajectory Warping for Contact-Rich Dexterous Manipulation Demonstrations

Reference implementation for the IROS 2026 submission *"Kinematic Non-Rigid
Spatio-Temporal Trajectory Warping for Contact-Rich Dexterous Manipulation
Demonstrations."*

`trajwarp` repurposes an existing contact-rich manipulation demonstration (hand
+ object trajectories with per-frame hand-object contact distributions) into new
variations that satisfy **spatial waypoints**, **environmental barriers**,
**temporal shifts**, and **new start/end configurations** — while preserving the
hand-object contact behavior of the original motion. It runs in minutes on a CPU
and generalizes across manipulators (human MANO hand, Allegro hand, Franka
gripper).

---

## Method overview

The method (paper §III) is an **object-centric** pipeline: first warp the object
trajectory to satisfy the inputs, then recover the hand trajectory from
hand-object contact distributions. There are four stages, and the code mirrors
them one-to-one:

| Paper stage | What it does | Module |
| --- | --- | --- |
| §III-A Spatial waypoint-constrained object warp | Warps the object path through specified positions/orientations (Algorithm 1, `TRANSFORMSEGMENT`) | [`trajwarp/object_warp/spatial.py`](trajwarp/object_warp/spatial.py) |
| §III-B Barrier-constrained object warp | Inflates barriers by the object's bounding ball (Minkowski sum) and projects the path out of intersections | [`trajwarp/object_warp/barriers.py`](trajwarp/object_warp/barriers.py) |
| §III-C Temporal waypoint-constrained object warp | Chord-length B-spline retiming realizing the `t → t'` timewarp | [`trajwarp/object_warp/smoothing.py`](trajwarp/object_warp/smoothing.py) |
| §III-D Hand warping | Per-timestep IK that reproduces the original contact correspondences and avoids barriers, then light smoothing | [`trajwarp/hand_warp/`](trajwarp/hand_warp/) |

The top-level driver [`trajwarp/pipeline.py`](trajwarp/pipeline.py) strings these
stages together:

```python
config = load_config(args.config)        # trajwarp.io.config
state  = load_demonstration(config)      # trajwarp.io.{spline_io,contact_io,scene_xml}
warp_object(state)                        # trajwarp.object_warp (§III-A/B/C)
warp_hand(state)                          # trajwarp.hand_warp.strategies.contact_based (§III-D)
metrics = compute_metrics(state)          # trajwarp.viz.metrics
save_outputs(state, metrics)
view_result(state)                        # trajwarp.viz.viewer
```

The hand-warp stage is pluggable: the contact-based method (ours) and two naive
SE(3) baselines all share the object warp and differ only in their
[`trajwarp/hand_warp/strategies/`](trajwarp/hand_warp/strategies/).

---

## Repository structure

```
trajectoryRetargeting/
├── retarget.py                 # our method (contact-based hand warp)
├── retarget_naive_spline.py    # baseline: keyframe + spline hand interpolation
├── visualize_original.py       # view the source demonstration
├── visualize_retargeted.py     # view a saved retargeted trajectory
├── evaluate.py                 # build the paper's comparison tables (Markdown/CSV)
├── save_initial.py             # convert raw demonstration splines to .npy
├── optimize_franka.py          # Franka-gripper generalization (paper §IV-C)
├── playback_franka.py          # replay a Franka result
│
├── trajwarp/                   # the library (mirrors paper §III)
│   ├── pipeline.py             # stage orchestration + WarpState
│   ├── io/                     # config, spline/contact parsing, MuJoCo XML
│   ├── object_warp/            # spatial, barriers, smoothing/temporal
│   ├── hand_warp/              # fk, optimizer, barrier SDF, contacts, strategies/
│   ├── viz/                    # interactive viewer + metrics
│   ├── geometry/               # procedural meshes (bins)
│   └── utils/                  # quaternion + agent (MANO/Allegro) helpers
│
├── scenes/                     # MuJoCo scene XMLs (kitchen, franka_in_kitchen)
├── agents/                     # hand models (MANO_right, Allegro_right)
├── tasks/                      # per-task scene fragments + object meshes
├── meshes/                     # object meshes (+ vendored kitchen assets, downloaded)
├── retargeting_configs/        # one JSON per warping trial + schema.json
└── scripts/                    # download_data.sh, reproduce_paper.sh
```

Large data (`startingTrajectories/`, vendored kitchen assets under
`meshes/robocasa` and `meshes/robosuite`, and `final_trajectories_*/` outputs)
is **git-ignored** and fetched separately — see [Data download](#2-data-download).

---

## Installation

### 1. Environment

Requires Python 3.10 and a working OpenGL stack for the MuJoCo viewer.

```bash
git clone https://github.com/hyojp13/trajectoryRetargeting.git
cd trajectoryRetargeting

conda create -n trajwarp python=3.10 -y
conda activate trajwarp
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used for the paper (MuJoCo 3.2.7,
PyTorch 2.6.0, SciPy 1.15.1, Open3D 0.19.0, ...). For GPU IK, install a CUDA
build of PyTorch and set `"optimization_device": "cuda"` in the config.

> **macOS only:** any command that opens the interactive viewer must be run with
> `mjpython` instead of `python` (a MuJoCo requirement for on-screen windows).
> This applies to `visualize_original.py`, `visualize_retargeted.py`,
> `playback_franka.py`, and `retarget*.py` / `optimize_franka.py` when *not* run
> with `--no-view`. Headless commands (`--no-view`, `evaluate.py`,
> `scripts/download_data.py`) use plain `python`. On Linux/Windows, use `python`
> everywhere. The `python` examples below are annotated accordingly.

### 2. Data download

The demonstration trajectories (~533 MB, derived from the GRAB dataset) and the
vendored RoboCasa/RoboSuite kitchen assets (~146 MB) are hosted externally:

```bash
bash scripts/download_data.sh
# or, without curl/wget:
python scripts/download_data.py
```

This populates `startingTrajectories/` and `meshes/{robocasa,robosuite}/`. Set
`STARTING_TRAJECTORIES_URL` / `KITCHEN_ASSETS_URL` to override the hosts.

Each demonstration lives at `startingTrajectories/<agent>/<task>/` and contains:

```
hand.smexp        # hand trajectory spline export
object.smexp      # object trajectory spline export
contacts.lcexp    # per-frame hand-object contact correspondences
```

where `<agent>` is `trajectories`/`MANO_right` (human MANO hand) or
`Allegro_right`.

---

## Quick start

Run a single warping trial and open the viewer (macOS: `mjpython`):

```bash
python retarget.py retargeting_configs/mug_pass_wall.json   # macOS: mjpython
```

Run headless (no viewer), which is what the batch scripts use:

```bash
python retarget.py retargeting_configs/mug_pass_wall.json --no-view
```

Outputs are written to `final_trajectories_<loss_threshold>/` as:

```
<name>_hand.npy       # (hand_dof, frames) wrist + joint trajectory
<name>_object.npy     # (frames, 7) object pose [x y z qw qx qy qz]
<name>_metrics.npy    # contact-distance metrics over the contact window
<name>_timewarp.npy   # timewarp + spline metadata
<name>_data.txt       # timing + hyperparameters
```

The baseline uses the same interface and writes a `_naive` suffix:

```bash
python retarget_naive_spline.py retargeting_configs/mug_pass_wall.json --no-view
```

---

## Visualizing

These open an interactive window, so on macOS run them with `mjpython` (Linux/Windows: `python`).

```bash
# The original demonstration (before warping), with contact points overlaid:
mjpython visualize_original.py retargeting_configs/mug_pass_wall.json

# A previously saved retargeted result:
mjpython visualize_retargeted.py retargeting_configs/mug_pass_wall.json
mjpython visualize_retargeted.py retargeting_configs/mug_pass_wall.json \
    --trajectory-dir final_trajectories_0.0035
```

The object path is colored along a green→red time gradient over the contact
window; barriers are drawn as translucent gray primitives.

---

## Reproducing the paper tables

`evaluate.py` reads the saved artifacts and reports, per trajectory: object-to-
wrist distance, contact distance (avg/max), contact-window length, object/hand
path-length ratios (vs. the initial demonstration), temporal-warp RMS
discrepancy, and retargeting time.

```bash
# Evaluate specific results in a directory:
python evaluate.py mug_pass_wall teapot_pour_cups --dir final_trajectories_0.0035

# Evaluate every config and write Markdown + CSV:
python evaluate.py --all --dir final_trajectories_0.0035 \
    --markdown results/table.md --csv results/table.csv
```

To regenerate everything from scratch (sweeps loss thresholds, runs all configs,
builds tables):

```bash
bash scripts/reproduce_paper.sh
# options:
THRESHOLDS="0.0035 0.01" bash scripts/reproduce_paper.sh   # custom threshold sweep
BASELINES=1 bash scripts/reproduce_paper.sh                # also run the naive baselines
```

---

## Configuration reference

Each warping trial is a JSON file in `retargeting_configs/`, validated against
[`retargeting_configs/schema.json`](retargeting_configs/schema.json) (parsed by
[`trajwarp/io/config.py`](trajwarp/io/config.py)). Example:

```json
{
  "agent": "trajectories",
  "task": "mug_pass",
  "scene_file": "scenes/kitchen2.xml",
  "object_mesh_file": "meshes/mug_pass_1_full_export_objectmesh.obj",
  "rotation": [0, 0, -101.386],

  "waypts": [[[0, 0, 0.1], 0.68, 0.75], [[0, 0, 0.15], 0.7, 0.84, [180, 0, 0]]],
  "new_start_pos_shift": [0, 0, 0],
  "end_final_pos_shift": [0, 0, 0],
  "end_obj_pos_shift": [0, 0, 0],
  "barriers": [["rect", {"dims": [0.7, 0.08, 1], "pos": [0.29, -1.75, 1.42]}]],

  "boundary_radius": 0.1,
  "hand_boundary_radius": 0.1,
  "learning_rate": 0.01,
  "n_iter": 1000,
  "first_frame_iter": 1000,
  "extra_pt_count": 0,
  "optimization_device": "cpu"
}
```

| Field | Meaning |
| --- | --- |
| `agent` | `trajectories`/`MANO_right` (MANO hand) or `Allegro_right` (Allegro hand) |
| `task` | demonstration under `startingTrajectories/<agent>/<task>/`; also selects `tasks/<task>.xml` |
| `scene_file` | MuJoCo scene the demonstration plays in |
| `object_mesh_file` | repo-relative path to the manipulated object's mesh |
| `rotation` | `[roll, pitch, yaw]` (deg) aligning the demonstration into the scene |
| `waypts` | list of `[position_shift, t_in, t_out]` (+ optional 4th `[r,p,y]`); `t_in`/`t_out` are normalized times in `[0,1]` |
| `new_start_pos_shift` / `end_final_pos_shift` / `end_obj_pos_shift` | translations of the object's start / terminal / placement pose |
| `barriers` | `["rect", {dims, pos}]` or `["sphere", {rad, pos}]` obstacles the object must avoid |
| `visuals` | optional cosmetic meshes `["mesh.obj", {pos, scale, rotation, rgba}]` (rendering only) |
| `boundary_radius` / `hand_boundary_radius` | barrier inflation radii for the object / hand segments |
| `learning_rate`, `n_iter`, `first_frame_iter`, `loss_threshold` | per-frame IK optimizer settings (§III-D) |
| `extra_pt_count` | extra interpolation points appended after the contact window |
| `optimization_device` | torch device for the IK: `cpu`, `cuda`, or `mps` |

A waypoint `w = (p, r, t, t')` corresponds directly to the paper's definition:
`p` is the world-space position the object must pass through, `r` the rotation
imposed over the preceding segment, `t` the time it occurs in the *input* and
`t'` the time it is moved to in the *output*.

---

## Generalization to other manipulators

The same pipeline handles different hands by switching `agent`; the scene is
assembled with the correct hand model automatically (`trajwarp/pipeline.py`
injects the agent's `assets/actuators/body` XML).

```bash
# Allegro hand (macOS: mjpython, since it opens the viewer):
python retarget.py retargeting_configs/apple_pass_allegro.json   # macOS: mjpython
```

A Franka parallel-jaw gripper variant (paper §IV-C) is provided as standalone
scripts (both open a viewer; macOS: `mjpython`):

```bash
python optimize_franka.py    # solve the Franka grasp/trajectory  (macOS: mjpython)
python playback_franka.py    # replay the saved Franka result     (macOS: mjpython)
```

---

## Adding a new demonstration

1. Place the demonstration under
   `startingTrajectories/<agent>/<your_task>/` with `hand.smexp`,
   `object.smexp`, and `contacts.lcexp`.
2. Add the object mesh to `meshes/` and (if needed) a `tasks/<your_task>.xml`
   scene fragment.
3. Author `retargeting_configs/<your_task>.json` (start by copying an existing
   config; it is validated against the schema on load).
4. Run it (macOS: `mjpython`, since this opens the viewer):

   ```bash
   python retarget.py retargeting_configs/<your_task>.json   # macOS: mjpython
   ```

---

## Citation

```bibtex
@inproceedings{trajwarp2026,
  title     = {Kinematic Non-Rigid Spatio-Temporal Trajectory Warping for
               Contact-Rich Dexterous Manipulation Demonstrations},
  author    = {Park, Hyojae and Lakshmipathy, Arjun S. and Pollard, Nancy S.},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

Demonstrations are derived from the **GRAB** dataset; contact correspondences
were extracted following the cited preprocessing pipeline. Please cite those
works as appropriate when using the data.

## License

Released under the MIT License.
