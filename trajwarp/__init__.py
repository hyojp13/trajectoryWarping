"""trajwarp: kinematic non-rigid spatio-temporal trajectory warping for
contact-rich dexterous manipulation demonstrations.

The package mirrors the four-stage pipeline described in the accompanying paper:

    object_warp.spatial    (Sec. III-A)  spatial waypoint-constrained warp
    object_warp.barriers   (Sec. III-B)  barrier-constrained warp
    object_warp.smoothing  (Sec. III-C)  temporal waypoint-constrained warp
    hand_warp              (Sec. III-D)  contact-driven hand recovery via IK

See ``retarget.py`` for the top-level driver that strings these stages together.
"""

__version__ = "1.0.0"
