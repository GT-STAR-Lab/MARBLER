from rps.utilities.controllers import *
import numpy as np

from utilities import *

class DemoPredatorAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.controller = create_clf_unicycle_position_controller()
    
    def getAction(self, observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        action = {}
        if observations['prey_loc'] == []:
            action['Velocity'] = (get_random_vel())
        else:
            goal = np.array(observations['prey_loc'])
            start = np.array([observations['agent_loc']]).T
            action['Velocity'] = self.controller(start, goal)
       
        self.actions.append(action)
        return action

class DemoCaptureAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.controller = create_clf_unicycle_position_controller()
    
    def getAction(self, observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        action = {}
        if observations['prey_loc'] == []:
            action['Velocity'] = (get_random_vel())
            action['Capture'] = False
        else:
            goal = np.array(observations['prey_loc'])
            start = np.array([observations['agent_loc']]).T
            action['Velocity'] = self.controller(start, goal)
            action['Capture'] = True #Obviously not ideal
       
        self.actions.append(action)
        return action
