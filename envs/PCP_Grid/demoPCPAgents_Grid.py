from rps.utilities.controllers import *
import numpy as np

from utilities import *

class DemoPredatorAgent:
    '''
    Naive predator agent that just drives straight until an agent sees the prey and then drives to it
    '''

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
            action['Velocity'] = 'right'
        else:
            goal = np.array(observations['prey_loc'])
            start = np.array([observations['agent_loc']]).T
            action['Velocity'] = 'stop'
       
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
    
    def getAction(self, observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        action = {}
        if observations['prey_loc'] == []:
            action['Velocity'] = 'right'
            action['Capture'] = False
        else:
            goal = np.array(observations['prey_loc'])
            start = np.array([observations['agent_loc']]).T
            action['Velocity'] = 'stop'
            action['Capture'] = True #Obviously not ideal. This should only be true when the prey is in the capture range of the agent
       
        self.actions.append(action)
        return action
