import numpy as np

PHASES = {
    "APPROACH": 0,
    "DESCEND": 1,
    "GRASP_SETTLE": 2,
    "LIFT": 3,
    "MOVE": 4,
    "FINE": 5,
}

def label_phase(
    state: np.ndarray,
    attach_dist_thresh: float = 0.055,
    descend_height_margin: float = 0.02,
    lift_height: float = 0.17,
    fine_radius: float = 0.020,
    fine_inner_radius: float = 0.018
):
    eef = state[0:3]
    grip = state[3]
    cube = state[4:7]
    tgt = state[7:9]
    horiz_err = np.linalg.norm(eef[:2] - cube[:2])
    vertical_gap = eef[2] - cube[2]
    dist_target = np.linalg.norm(cube[:2] - tgt)
    dist_eef_cube = np.linalg.norm(eef - cube)
    attached = (grip > 0.5) and (dist_eef_cube < attach_dist_thresh)
    if not attached:
        if grip < 0.5:
            if horiz_err > 0.02:
                return PHASES["APPROACH"]
            if vertical_gap > descend_height_margin:
                return PHASES["APPROACH"]
            return PHASES["DESCEND"]
        else:
            if vertical_gap <= descend_height_margin and horiz_err < 0.012:
                return PHASES["GRASP_SETTLE"]
            return PHASES["DESCEND"]
    if eef[2] < lift_height:
        return PHASES["LIFT"]
    if dist_target <= fine_inner_radius:
        return PHASES["FINE"]
    if dist_target <= fine_radius:
        return PHASES["MOVE"]
    return PHASES["MOVE"]