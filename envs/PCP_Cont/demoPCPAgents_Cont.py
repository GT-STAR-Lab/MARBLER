from rps.utilities.controllers import *
import numpy as np

from utilities import *

def getNeighborsPrey(observations):
    for i in range(3, len(observations), 7):
            if (observations[i:i+2] != [-1,-1]).all():
                return observations[i:i+2]
    return np.array([-1,-1])

class DemoPredatorAgent:
    '''
    Naive predator agent that just drives straight until an agent sees the prey and then drives to it
    '''

    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.controller = create_clf_unicycle_position_controller()
    
    def getAction(self, observations, critic_observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        prey_loc = getNeighborsPrey(observations)
        stop_dist = observations[5]

        if (prey_loc == [-1,-1]).all():
            action = get_random_vel()
        elif np.linalg.norm(observations[:2] - prey_loc) > stop_dist:
            goal = np.array(prey_loc)[np.newaxis].T
            start = np.array([observations[:3]]).T
            action = self.controller(start, goal).T[0]
        else:
             action = np.array([0,0]).T
           
        self.actions.append(action)
        return action

class DemoCaptureAgent:
    '''
    Naive capture agent that just drives straight until an agent sees the prey and then drives to it
    Tries to capture the prey each time step that after an agent has seen it, occuring big penalties
    '''

    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.controller = create_clf_unicycle_position_controller()
    
    def getAction(self, observations, critic_observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        capture_dist = observations[6]
        prey_loc = getNeighborsPrey(observations)
        if (prey_loc == [-1,-1]).all():
            action = get_random_vel()
        elif np.linalg.norm(observations[:2] - prey_loc) > capture_dist:
            goal = np.array(prey_loc)[np.newaxis].T
            start = np.array([observations[:3]]).T
            action = self.controller(start, goal).T[0]
        else: #Making 0,0 implicitly capture the prey
            action = np.array([0,0]).T
       
        self.actions.append(action)
        return action
