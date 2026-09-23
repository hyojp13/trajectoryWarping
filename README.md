# trajwarp — Kinematic Non-Rigid Spatio-Temporal Trajectory Warping for Contact-Rich Dexterous Manipulation Demonstrations

Code for the IROS 2026 paper *Kinematic Non-Rigid Spatio-Temporal Trajectory
Warping for Contact-Rich Dexterous Manipulation Demonstrations*.

Given one demonstration of a hand manipulating an object (hand and object
trajectories plus per-frame hand-object contacts), trajwarp produces new
versions of it that pass through spatial waypoints, avoid barriers in the
scene, shift events in time, or start and end somewhere else. The hand keeps
the same contacts with the object as in the original motion. A warp takes a few
minutes on a CPU, and the same code works for the MANO hand, the Allegro hand,
and a Franka gripper.

---

## Method overview

We warp the object trajectory first, then solve for the hand from the contacts.
Each stage in Section III of the paper has its own module:

| Paper stage | What it does | Module |
| --- | --- | --- |
| §III-A Spatial waypoint-constrained object warp | Warps the object path through specified positions/orientations (Algorithm 1, `TRANSFORMSEGMENT`) | [`trajwarp/object_warp/spatial.py`](trajwarp/object_warp/spatial.py) |
| §III-B Barrier-constrained object warp | Inflates barriers by the object's bounding ball (Minkowski sum) and projects the path out of intersections | [`trajwarp/object_warp/barriers.py`](trajwarp/object_warp/barriers.py) |
| §III-C Temporal waypoint-constrained object warp | Chord-length B-spline retiming realizing the `t → t'` timewarp | [`trajwarp/object_warp/smoothing.py`](trajwarp/object_warp/smoothing.py) |
| §III-D Hand warping | Per-timestep IK that reproduces the original contact correspondences and avoids barriers, then light smoothing | [`trajwarp/hand_warp/`](trajwarp/hand_warp/) |

[`trajwarp/pipeline.py`](trajwarp/pipeline.py) runs the stages in order:

```python
config = load_config(args.config)        # trajwarp.io.config
state  = load_demonstration(config)      # trajwarp.io.{spline_io,contact_io,scene_xml}
warp_object(state)                        # trajwarp.object_warp (§III-A/B/C)
warp_hand(state)                          # trajwarp.hand_warp.strategies.contact_based (§III-D)
metrics = compute_metrics(state)          # trajwarp.viz.metrics
save_outputs(state, metrics)
view_result(state)                        # trajwarp.viz.viewer
```

`warp_hand` comes from [`trajwarp/hand_warp/strategies/`](trajwarp/hand_warp/strategies/).
Our contact-based method is `contact_based.py`. The baseline in
`naive_spline.py` uses the same object warp and interpolates the hand between
keyframes with a spline.

---

## Repository structure

```
trajectoryWarping/
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
│   ├── io/                     # config, spline/contact parsing, MuJoCo XML, data check
│   ├── object_warp/            # spatial, barriers, smoothing/temporal
│   ├── hand_warp/              # fk, optimizer, barrier SDF, contacts, strategies/
│   ├── viz/                    # interactive viewer + metrics
│   ├── geometry/               # procedural meshes (bins)
│   └── utils/                  # quaternion + agent (MANO/Allegro) helpers
│
├── scenes/                     # MuJoCo scene XMLs (kitchen, franka_in_kitchen)
├── agents/                     # hand models (MANO_right, Allegro_right)
├── tasks/                      # per-task scene fragments
├── meshes/                     # barrier meshes (+ object meshes and kitchen assets, not tracked)
├── retargeting_configs/        # one JSON per warping trial + schema.json
└── scripts/                    # download_data.sh, reproduce_paper.sh
```

---

## Installation

### 1. Environment

You need Python 3.10 and OpenGL for the MuJoCo viewer.

```bash
git clone https://github.com/hyojp13/trajectoryWarping.git
cd trajectoryWarping

conda create -n trajwarp python=3.10 -y
conda activate trajwarp
pip install -r requirements.txt
```

`requirements.txt` pins the versions we used for the paper (MuJoCo 3.2.7,
PyTorch 2.6.0, SciPy 1.15.1, Open3D 0.19.0, and others). To run the IK on a GPU,
install a CUDA build of PyTorch and set `"optimization_device": "cuda"` in the
config.

> **macOS only:** MuJoCo can only open a viewer window from `mjpython`, so use
> `mjpython` in place of `python` for `visualize_original.py`,
> `visualize_retargeted.py`, `playback_franka.py`, and for `retarget*.py` or
> `optimize_franka.py` without `--no-view`. Everything else, and everything on
> Linux or Windows, runs with plain `python`.

### 2. Data

This repository contains the code, the hand models, the configs, and the
barrier meshes. You have to get the rest yourself.

**Kitchen assets.** The RoboCasa/RoboSuite meshes and textures for the kitchen
scene (~146 MB) are downloaded separately:

```bash
bash scripts/download_data.sh
# or, without curl/wget:
python scripts/download_data.py
```

They go into `meshes/{robocasa,robosuite}/`. Set `KITCHEN_ASSETS_URL` to
download from somewhere else.

**Demonstrations, object meshes, and MANO hand meshes.** The demonstrations and
object meshes come from the [GRAB](https://grab.is.tue.mpg.de) dataset, and the
human hand meshes from [MANO](https://mano.is.tue.mpg.de). Neither license
allows us to redistribute them (see [Data licensing](#data-licensing)), so you
need to register for both and prepare the files yourself. `retarget.py`,
`retarget_naive_spline.py`, and `visualize_original.py` check for these files
on startup and print a list of anything missing.

| What | Where it goes | Source |
| --- | --- | --- |
| Demonstrations | `startingTrajectories/<agent>/<task>/{hand.smexp, object.smexp, contacts.lcexp}` | GRAB sequences, converted |
| Object meshes | `meshes/<name>.obj` (the `object_mesh_file` in each config) | GRAB object meshes |
| MANO hand meshes | `agents/MANO_right/geom_assets/right_<link>.obj` (16 files, listed below) | MANO right hand, split per link |

Each demonstration directory contains:

```
hand.smexp        # hand trajectory spline export
object.smexp      # object trajectory spline export
contacts.lcexp    # per-frame hand-object contact correspondences
```

where `<agent>` is `trajectories`/`MANO_right` (human MANO hand) or
`Allegro_right`. The Allegro model and its meshes are included in the
repository.

The bundled configs use these demonstrations and object meshes:

| Configs | `startingTrajectories/…` | Object mesh |
| --- | --- | --- |
| `fryingpan_*` | `trajectories/fryingpan_cook/` | `meshes/fryingpan_cook_2_full_export_objectmesh.obj` |
| `mug_pass*` | `trajectories/mug_pass/` | `meshes/mug_pass_1_full_export_objectmesh.obj` |
| `teapot_pour_*` | `trajectories/teapot_pour/` | `meshes/teapot_pour_1_full_export_objectmesh.obj` |
| `waterbottle_shake*` | `trajectories/waterbottle_shake/` | `meshes/waterbottle_shake_1_full_export_objectmesh.obj` |
| `cube_messy*` | `trajectories/cube/` | `meshes/cubemedium_inspect_1_full_export_objectmesh.obj` |
| `apple_pass_allegro` | `Allegro_right/apple_pass/` | `meshes/apple.obj` |

**MANO hand meshes.** The MuJoCo MANO hand (`agents/MANO_right/assets.xml`) and
the contact loader (`trajwarp/io/contact_io.py`) expect one mesh per hand link:

```
agents/MANO_right/geom_assets/
  right_wrist.obj
  right_thumb1.obj   right_thumb2.obj   right_thumb3.obj
  right_index1.obj   right_index2.obj   right_index3.obj
  right_middle1.obj  right_middle2.obj  right_middle3.obj
  right_ring1.obj    right_ring2.obj    right_ring3.obj
  right_pinky1.obj   right_pinky2.obj   right_pinky3.obj
```

`contacts.lcexp` stores hand contacts as vertex indices into these meshes, so
they have to be the same meshes the contacts were extracted on. A per-link
split with a different vertex order will load without errors and give wrong
contacts. The MANO release only includes the parametric model
(`MANO_RIGHT.pkl`), so please get in touch with us if you want to reproduce
our split.

If you have access to a licensed copy of our preprocessed demonstrations,
`STARTING_TRAJECTORIES_URL=<url> bash scripts/download_data.sh` downloads and
unpacks it.

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

The baseline takes the same arguments and adds a `_naive` suffix to its
outputs:

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

The object path is drawn in a gradient from green to red over the contact
window, and barriers appear as translucent gray boxes and spheres.

---

## Reproducing the paper tables

`evaluate.py` reads saved results and reports, for each trajectory, the
object-to-wrist distance, average and maximum contact distance, contact window
length, object and hand path length relative to the input, RMS temporal-warp
error, and run time.

```bash
# Evaluate specific results in a directory:
python evaluate.py mug_pass_wall teapot_pour_cups --dir final_trajectories_0.0035

# Evaluate every config and write Markdown + CSV:
python evaluate.py --all --dir final_trajectories_0.0035 \
    --markdown results/table.md --csv results/table.csv
```

`scripts/reproduce_paper.sh` runs every config at each loss threshold and
builds the tables:

```bash
bash scripts/reproduce_paper.sh
# options:
THRESHOLDS="0.0035 0.01" bash scripts/reproduce_paper.sh   # custom threshold sweep
BASELINES=1 bash scripts/reproduce_paper.sh                # also run the naive baseline
```

---

## Configuration reference

Each warping trial is a JSON file in `retargeting_configs/`.
[`trajwarp/io/config.py`](trajwarp/io/config.py) checks it against
[`retargeting_configs/schema.json`](retargeting_configs/schema.json) on load.
Example:

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

A waypoint `w = (p, r, t, t')` matches the definition in the paper: the object
passes through position `p` with rotation `r` applied over the segment before
it, and the event at time `t` in the input happens at `t'` in the output.

---

## Generalization to other manipulators

Set `agent` to switch hands. `trajwarp/pipeline.py` swaps the matching
`assets`, `actuators`, and `body` XML into the scene.

```bash
# Allegro hand (macOS: mjpython, since it opens the viewer):
python retarget.py retargeting_configs/apple_pass_allegro.json   # macOS: mjpython
```

The Franka gripper experiment (paper §IV-C) has its own scripts, both of which
open a viewer (macOS: `mjpython`):

```bash
python optimize_franka.py    # solve the Franka grasp/trajectory  (macOS: mjpython)
python playback_franka.py    # replay the saved Franka result     (macOS: mjpython)
```

---

## Adding a new demonstration

1. Put `hand.smexp`, `object.smexp`, and `contacts.lcexp` in
   `startingTrajectories/<agent>/<your_task>/`.
2. Add the object mesh to `meshes/`, and a `tasks/<your_task>.xml` scene
   fragment if the task needs one.
3. Copy an existing config to `retargeting_configs/<your_task>.json` and edit
   it. It is checked against the schema when it loads.
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

If you use the demonstration data, please also cite GRAB and the papers listed
under [Data licensing](#data-licensing).

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)).

### Data licensing

The MIT License does not cover the data. Our demonstrations, contact
correspondences, and object meshes are derived from **GRAB** (Taheri et al.,
ECCV 2020), whose objects come from ContactDB (Brahmbhatt et al., CVPR 2019).
The hand meshes are derived from **MANO** (Romero et al., SIGGRAPH Asia 2017).
The Max Planck Institute for Intelligent Systems licenses GRAB and MANO for
non-commercial research only and does not allow redistribution, which is why
the files are not in this repository. You can request access at
[grab.is.tue.mpg.de](https://grab.is.tue.mpg.de) and
[mano.is.tue.mpg.de](https://mano.is.tue.mpg.de).
