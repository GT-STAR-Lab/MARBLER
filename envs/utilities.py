import numpy as np

def is_close( agent_pose, prey_loc, sensing_radius):
    agent_position = agent_pose[:2].T
    return np.linalg.norm(agent_position - prey_loc, 2) <= sensing_radius