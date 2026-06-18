"""Quaternion conversions shared across the pipeline.

Two conventions appear throughout the codebase:

* MuJoCo stores quaternions as ``[w, x, y, z]``.
* SciPy (``scipy.spatial.transform.Rotation``) uses ``[x, y, z, w]``.

The ``convert_to_quaternions_*`` helpers turn the Euler-angle spline outputs
(produced by :mod:`trajwarp.io.spline_io`) into MuJoCo ``qpos`` arrays.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_to_quaternions_MANO(qpos_spline_data):
    """Convert a 51-DOF MANO Euler spline (3 translation + 16x3 rotations) into
    a 67-row MuJoCo ``qpos`` array (3 translation + 16 quaternions)."""
    qpos = np.zeros((67, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        for i in range(3, 51, 3):
            rotation = R.from_euler('xyz', qpos_spline_data[i:i + 3, j], degrees=False)
            qpos[int((i / 3 - 1) * 4 + 4): int(i / 3 * 4 + 3), j] = rotation.as_quat()[:3]
            qpos[int((i / 3 - 1) * 4 + 3), j] = rotation.as_quat()[3]

    return qpos


def convert_to_quaternions_Allegro(qpos_spline_data):
    """Convert a 22-DOF Allegro Euler spline into a 23-row MuJoCo ``qpos`` array
    (3 translation + wrist quaternion + 16 finger joints)."""
    qpos = np.zeros((23, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]
    qpos[7:, :] = qpos_spline_data[6:, :]

    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]

    return qpos


def convert_to_quaternions_object(qpos_spline_data):
    """Convert a 6-DOF object Euler spline into a 7-row MuJoCo pose array
    (3 translation + quaternion)."""
    qpos = np.zeros((7, qpos_spline_data.shape[1]))
    qpos[:3, :] = qpos_spline_data[:3, :]

    for j in range(qpos_spline_data.shape[1]):
        rotation = R.from_euler('xyz', qpos_spline_data[3:6, j], degrees=False)
        qpos[4:7, j] = rotation.as_quat()[:3]
        qpos[3, j] = rotation.as_quat()[3]

    return qpos


def rotate_keyframe_angles(keyframes, rotation):
    """Pre-compose ``rotation`` (degrees, applied in world frame) onto the
    existing Euler orientations stored in rows 3:6 of ``keyframes``.

    keyframes: (>=6, n_frames) array; rotation: [rx, ry, rz] in degrees.
    """
    keyframes_modified = keyframes.copy()

    additional_rot = R.from_euler('xyz', np.radians(rotation))

    for i in range(keyframes.shape[1]):
        existing_euler = keyframes[3:6, i]
        existing_rot = R.from_euler('xyz', existing_euler)

        combined_rot = additional_rot * existing_rot

        keyframes_modified[3:6, i] = combined_rot.as_euler('xyz')

    return keyframes_modified


def mujoco_to_scipy_quat(q):
    """``[w, x, y, z]`` -> ``[x, y, z, w]``."""
    return np.array([q[1], q[2], q[3], q[0]])


def scipy_to_mujoco_quat(q):
    """``[x, y, z, w]`` -> ``[w, x, y, z]``."""
    return np.array([q[3], q[0], q[1], q[2]])
