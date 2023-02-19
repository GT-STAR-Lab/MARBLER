import numpy as np
import random

def is_close( agent_poses, agent_index, prey_loc, sensing_radius):
    agent_pose = agent_poses[:2, agent_index]
    prey_loc = prey_loc.reshape((1,2))[0]
    # print(agent_pose, prey_loc, np.linalg.norm(agent_pose - prey_loc), sensing_radius)
    return np.linalg.norm(agent_pose - prey_loc) <= sensing_radius

def get_random_vel():
    '''
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second (range +- .1-)
    '''
    linear_vel = random.uniform(0.05,0.2) 
    #* (1 if random.uniform(-1,1)>0 else -1)
    angular_vel = 0
    # angular_vel = random.uniform(0,10)
    return np.array([linear_vel,angular_vel]).T
