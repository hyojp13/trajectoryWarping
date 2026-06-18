"""Retargeting configuration loading and validation.

A configuration is a JSON file describing one warping trial: which demonstration
to start from, the scene, spatio-temporal waypoints, barriers, terminal
position shifts, and optimizer settings. See ``retargeting_configs/schema.json``
for the full schema and ``retargeting_configs/*.json`` for examples.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

# Fields that must be present in every config file.
REQUIRED_FIELDS = [
    'agent', 'task', 'scene_file', 'barriers',
    'learning_rate', 'n_iter', 'first_frame_iter',
    'boundary_radius', 'hand_boundary_radius',
    'new_start_pos_shift', 'end_final_pos_shift', 'end_obj_pos_shift',
    'waypts', 'extra_pt_count', 'optimization_device',
    'object_mesh_file', 'rotation',
]


@dataclass
class RetargetConfig:
    """Validated, parsed view of a retargeting config file."""

    agent: str
    task: str
    scene_file: str
    object_mesh_file: str

    # Optimizer
    learning_rate: float
    n_iter: int
    first_frame_iter: int
    optimization_device: str
    loss_threshold: float

    # Object/hand trajectory constraints
    boundary_radius: float
    hand_boundary_radius: float
    new_start_pos_shift: np.ndarray
    end_final_pos_shift: np.ndarray
    end_obj_pos_shift: np.ndarray
    waypts: list
    extra_pt_count: int
    rotation: list

    # Scene contents
    barriers: list
    visuals: list

    # Hand-barrier loss (only used when barrier optimization is enabled)
    barrier_weight: float
    barrier_margin: float
    barrier_n: float

    # The unparsed dict, for forward compatibility / debugging.
    raw: dict = field(default_factory=dict, repr=False)


def _parse_waypoints(raw_waypts):
    """Convert ``[[pos, t, t'], ...]`` (optionally with a 4th rotation element)
    into tuples with ``pos`` as a numpy array, preserving the original layout."""
    waypts = []
    for w in raw_waypts:
        if len(w) == 4:
            waypts.append((np.array(w[0]), w[1], w[2], w[3]))
        else:
            waypts.append((np.array(w[0]), w[1], w[2]))
    return waypts


def _validate_against_schema(raw, config_path):
    """Validate ``raw`` against ``schema.json`` (next to the config) using
    ``jsonschema`` when both are available. Falls back silently to the
    required-field check below when either is missing, so the repo works
    without the optional ``jsonschema`` dependency installed."""
    schema_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), 'schema.json')
    if not os.path.exists(schema_path):
        return
    try:
        import jsonschema
    except ImportError:
        return
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=raw, schema=schema)
    except jsonschema.ValidationError as e:
        location = "/".join(str(p) for p in e.absolute_path) or "<root>"
        raise ValueError(
            f"Invalid config '{config_path}' at '{location}': {e.message}") from None


def load_config(config_path):
    """Load, validate, and parse a retargeting config JSON file."""
    with open(config_path, 'r') as f:
        raw = json.load(f)

    _validate_against_schema(raw, config_path)

    for field_name in REQUIRED_FIELDS:
        if field_name not in raw:
            raise ValueError(f"Configuration file must contain '{field_name}' field")

    return RetargetConfig(
        agent=raw['agent'],
        task=raw['task'],
        scene_file=raw['scene_file'],
        object_mesh_file=raw['object_mesh_file'],
        learning_rate=raw['learning_rate'],
        n_iter=raw['n_iter'],
        first_frame_iter=raw['first_frame_iter'],
        optimization_device=raw['optimization_device'],
        loss_threshold=raw.get('loss_threshold', 0.0035),
        boundary_radius=raw['boundary_radius'],
        hand_boundary_radius=raw['hand_boundary_radius'],
        new_start_pos_shift=np.array(raw['new_start_pos_shift']),
        end_final_pos_shift=np.array(raw['end_final_pos_shift']),
        end_obj_pos_shift=np.array(raw['end_obj_pos_shift']),
        waypts=_parse_waypoints(raw['waypts']),
        extra_pt_count=raw['extra_pt_count'],
        rotation=raw['rotation'],
        barriers=raw['barriers'],
        visuals=raw.get('visuals', []),
        barrier_weight=raw.get('barrier_weight', 0.6),
        barrier_margin=raw.get('barrier_margin', 0.01),
        barrier_n=raw.get('barrier_n', 2.0),
        raw=raw,
    )
