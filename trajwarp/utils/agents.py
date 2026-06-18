"""End-effector registry.

Maps a config ``agent`` string to its hand model family and the ordered list of
MuJoCo body names that make up the articulated hand. The order matters: it
defines the component indexing used when loading per-link contact data and when
running forward kinematics.
"""

# MuJoCo body names for the MANO right hand, in component order
# (wrist first, then thumb/ring/pinky/middle/index phalanges).
MANO_BODY_NAMES = [
    'wrist',
    'thumb1', 'thumb2', 'thumb3',
    'ring1', 'ring2', 'ring3',
    'pinky1', 'pinky2', 'pinky3',
    'middle1', 'middle2', 'middle3',
    'index1', 'index2', 'index3',
]

# MuJoCo body names for the Allegro right hand, in component order
# (palm first, then thumb/first/middle/ring finger links).
ALLEGRO_BODY_NAMES = [
    'allegro_palm',
    'allegro_th_base', 'allegro_th_proximal', 'allegro_th_medial', 'allegro_th_distal', 'allegro_th_tip',
    'allegro_ff_base', 'allegro_ff_proximal', 'allegro_ff_medial', 'allegro_ff_distal', 'allegro_ff_tip',
    'allegro_mf_base', 'allegro_mf_proximal', 'allegro_mf_medial', 'allegro_mf_distal', 'allegro_mf_tip',
    'allegro_rf_base', 'allegro_rf_proximal', 'allegro_rf_medial', 'allegro_rf_distal', 'allegro_rf_tip',
]

# Config ``agent`` strings that resolve to the MANO hand. ``trajectories`` is the
# kitchen-demo data namespace, which also uses the MANO hand model.
_MANO_AGENTS = {'MANO_right', 'trajectories'}
_ALLEGRO_AGENTS = {'Allegro_right'}


def get_hand_type(agent):
    """Return ``'MANO'`` or ``'Allegro'`` for a config ``agent`` string."""
    if agent in _MANO_AGENTS:
        return 'MANO'
    if agent in _ALLEGRO_AGENTS:
        return 'Allegro'
    raise ValueError(f"Unknown agent '{agent}'")


def get_hand_body_names(agent):
    """Return the ordered list of MuJoCo body names for the agent's hand."""
    return ALLEGRO_BODY_NAMES if get_hand_type(agent) == 'Allegro' else MANO_BODY_NAMES
