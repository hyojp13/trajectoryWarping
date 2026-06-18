import numpy as np


def process_contacts(contacts_lcexp, hand_components_len):
    """
    Process contacts from .lcexp format to playTrajectory format.

    Args:
        contacts_lcexp: List of frame contacts from .lcexp file
                       Each frame contains: List of (obj_vertex, hand_link_idx, (face_idx, bary1, bary2, bary3))
        hand_components_len: Number of hand components (default 16)

    Returns:
        tuple: (hand_contacts, object_contacts)
            - hand_contacts: dict of hand component id -> list of contacts per frame
            - object_contacts: list of object vertex indices per frame
    """
    frames = len(contacts_lcexp)

    # Initialize hand and object contacts
    hand_contacts = {}  # per hand component, each contains a list of contacts per frame
    for i in range(hand_components_len):
        hand_contacts[i] = [None] * frames

    object_contacts = [None] * frames

    # Convert contacts from .lcexp format to playTrajectory format
    for frame_idx in range(frames):
        frame_contacts = contacts_lcexp[frame_idx]  # List of (obj_vertex, hand_link_idx, (face_idx, bary1, bary2, bary3))

        if len(frame_contacts) == 0:
            object_contacts[frame_idx] = np.array([], dtype=np.int64)
            continue

        # Extract object vertex indices for this frame
        obj_vertex_indices = []
        for contact in frame_contacts:
            obj_vertex_idx = contact[0]
            obj_vertex_indices.append(obj_vertex_idx)

        object_contacts[frame_idx] = np.array(obj_vertex_indices, dtype=np.int64)

        # Create mapping from object vertex index to index in object_contacts array
        obj_vertex_to_idx = {v: i for i, v in enumerate(obj_vertex_indices)}

        # Group contacts by hand component
        contacts_by_component = {}
        for contact in frame_contacts:
            obj_vertex_idx, hand_link_idx, hand_contact_info = contact
            face_idx, bary1, bary2, bary3 = hand_contact_info

            hand_component_id = hand_link_idx

            if hand_component_id < 0 or hand_component_id >= hand_components_len:
                raise Exception("Hand component id is out of range at frame " + str(frame_idx))

            # Get object_contact_idx (index into object_contacts[frame_idx])
            object_contact_idx = obj_vertex_to_idx[obj_vertex_idx]

            contact_tuple = (face_idx, (bary1, bary2, bary3), object_contact_idx)

            if hand_component_id not in contacts_by_component:
                contacts_by_component[hand_component_id] = []
            contacts_by_component[hand_component_id].append(contact_tuple)

        # Assign to hand_contacts
        for hand_component_id in contacts_by_component:
            hand_contacts[hand_component_id][frame_idx] = contacts_by_component[hand_component_id]

    # check that all frames have all object contacts covered
    for i in range(frames):
        contact_count = len(object_contacts[i])
        contact_checker = [False] * contact_count

        hand_contact_count = 0
        for j in range(hand_components_len):
            component_contacts = hand_contacts[j][i]
            if component_contacts is None:
                continue
            for k in component_contacts:
                contact_checker[k[2]] = True
                hand_contact_count += 1

        if False in contact_checker:
            raise Exception("Missing contact at frame " + str(i))
        if hand_contact_count > contact_count:
            raise Exception("More hand contact count than object contact count at frame " + str(i))

    return hand_contacts, object_contacts
