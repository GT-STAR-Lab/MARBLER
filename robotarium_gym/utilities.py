import numpy as np
import random

def is_close( agent_poses, agent_index, prey_loc, sensing_radius):
    agent_pose = agent_poses[:2, agent_index]
    prey_loc = prey_loc.reshape((1,2))[0]
    dist = np.linalg.norm(agent_pose - prey_loc)
    return dist <= sensing_radius, dist

def get_nearest_neighbors(poses, agent, num_neighbors):
    N = poses.shape[1]
    agents = np.arange(N)
    dists = [np.linalg.norm(poses[:2,x]-poses[:2,agent]) for x in agents]
    mins = np.argpartition(dists, num_neighbors+1)
    return np.delete(mins, np.argwhere(mins==agent))[:num_neighbors]


def get_random_vel():
    '''
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
    '''
    linear_vel = random.uniform(0.05,0.2) 
    #* (1 if random.uniform(-1,1)>0 else -1)
    angular_vel = 0
    # angular_vel = random.uniform(0,10)
    return np.array([linear_vel,angular_vel]).T

def convert_to_robotarium_poses(locations):
    poses = np.array(locations)
    N = poses.shape[0]
    return np.vstack((poses.T, np.zeros(N)))
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def generate_locations(args, num_robots, left = None, right = None,\
                    robotarium_poses = True, min_dist = None ):
    '''
        Generates random locations on a grid defined within the specified thresholds.
        Overlays a grid of unit size min_dist and picks num_robots random points from it 
    '''
    if left  == None: left  = args.LEFT 
    if right == None: right = args.RIGHT
    up   = args.UP
    down = args.DOWN
    if min_dist == None: min_dist = args.MIN_DIST
    buffer = args.MIN_DIST

    # overlay a grid over the allowed region
    cols = int( round( (right - left )/min_dist, 0)) - 1
    rows = int( round( (down - up )/min_dist, 0)) - 1

    # pick random locations from the grid
    grid_indices = np.random.choice( rows*cols, num_robots, replace = False)
    # convert grid locations back to robotarium coordinates
    locations = []
    for loc in grid_indices:
        locations.append([left + buffer + ( (loc % cols) * min_dist ),\
                            up + buffer + ( int(loc / cols) * min_dist ) ])

    if robotarium_poses:
        return convert_to_robotarium_poses(locations)

    return locations