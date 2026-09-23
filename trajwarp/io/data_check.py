"""Check that the data a config needs is on disk before running.

The GRAB demonstrations, GRAB object meshes, and MANO hand meshes can't be
redistributed, and the kitchen assets are downloaded separately, so a fresh
clone is missing all of them. Checking up front lets us list every missing file
at once instead of failing on the first one somewhere inside the pipeline.
"""
import os
import re

from trajwarp.utils.agents import MANO_BODY_NAMES, get_hand_type

README_DATA_URL = "https://github.com/hyojp13/trajectoryWarping#2-data"

# Per-link MANO meshes (MuJoCo hand model + contact loader); contacts.lcexp
# indexes vertices of these exact meshes.
MANO_MESH_DIR = "agents/MANO_right/geom_assets"
MANO_MESH_FILES = [f"right_{name}.obj" for name in MANO_BODY_NAMES]

DEMONSTRATION_FILES = ["hand.smexp", "object.smexp", "contacts.lcexp"]


class MissingDataError(FileNotFoundError):
    """Raised when data required by a config is not present on disk."""


def _missing(paths):
    return [p for p in paths if not os.path.exists(p)]


def check_required_data(config):
    """Raise ``MissingDataError`` if any file the config needs is missing.

    MANO meshes are only required for MANO agents, and kitchen assets only if
    the scene XML references them outside of comments.
    """
    groups = []

    demo_dir = f"startingTrajectories/{config.agent}/{config.task}"
    missing = _missing([f"{demo_dir}/{f}" for f in DEMONSTRATION_FILES])
    if missing:
        groups.append(("Demonstration data (derived from GRAB, not distributed)",
                       missing))

    missing = _missing([config.object_mesh_file])
    if missing:
        groups.append(("Object mesh (GRAB object, not distributed)", missing))

    if get_hand_type(config.agent) == "MANO":
        missing = _missing([f"{MANO_MESH_DIR}/{f}" for f in MANO_MESH_FILES])
        if missing:
            groups.append(("Per-link MANO hand meshes (MANO license, not distributed)",
                           missing))

    try:
        with open(config.scene_file) as f:
            scene = re.sub(r"<!--.*?-->", "", f.read(), flags=re.S)
    except OSError:
        groups.append(("Scene file", [config.scene_file]))
        scene = ""
    kitchen = [f"meshes/{lib}" for lib in ("robocasa", "robosuite")
               if f'file="{lib}/' in scene]
    missing = _missing(kitchen)
    if missing:
        groups.append(("Kitchen assets (run: bash scripts/download_data.sh)",
                       missing))

    if not groups:
        return

    lines = ["Missing data for this config:"]
    for title, paths in groups:
        lines.append(f"\n  {title}:")
        if len(paths) > 6:
            lines += [f"    {p}" for p in paths[:5]]
            lines.append(f"    ... and {len(paths) - 5} more")
        else:
            lines += [f"    {p}" for p in paths]
    lines.append(f"\nThe Data section of the README explains where these come "
                 f"from: {README_DATA_URL}")
    raise MissingDataError("\n".join(lines))
