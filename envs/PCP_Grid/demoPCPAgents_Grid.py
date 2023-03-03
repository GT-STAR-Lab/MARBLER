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
        stop_dist = .3 #Ideally this is learned in a real agent
        action = {}
        if observations['prey_loc'] == []:
            if observations['agent_loc'][0] < 1.2:
                action['Velocity'] = 'right'
            elif observations['agent_loc'][1] > -.8:
                action['Velocity'] = 'up'
            else:
                action['Velocity'] = 'down'
        else:                
            if observations['agent_loc'][1] < observations['prey_loc'][1] - stop_dist:
                action['Velocity'] = 'down'
            elif observations['agent_loc'][0] < observations['prey_loc'][0] - stop_dist:
                action['Velocity'] = 'right'
            elif observations['agent_loc'][0] > observations['prey_loc'][0] + stop_dist:
                action['Velocity'] = 'left'
            elif observations['agent_loc'][1] > observations['prey_loc'][1] + stop_dist:
                action['Velocity'] = 'up'
            else:
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
        stop_dist = .2

        if observations['prey_loc'] == []:
            if observations['agent_loc'][0] < 1.2:
                action['Velocity'] = 'right'
            elif observations['agent_loc'][1] > -.8:
                action['Velocity'] = 'up'
            else:
                action['Velocity'] = 'down'
            action['Capture'] = False
        else:
            if observations['agent_loc'][1] < observations['prey_loc'][1] - stop_dist:
                action['Velocity'] = 'down'
                action['Capture'] = False
            elif observations['agent_loc'][0] < observations['prey_loc'][0] - stop_dist:
                action['Velocity'] = 'right'
                action['Capture'] = False
            elif observations['agent_loc'][0] > observations['prey_loc'][0] + stop_dist:
                action['Velocity'] = 'left'
                action['Capture'] = False
            elif observations['agent_loc'][1] > observations['prey_loc'][1] + stop_dist:
                action['Velocity'] = 'up'
                action['Capture'] = False
            else:
                action['Velocity'] = 'stop'
                action['Capture'] = True 

        self.actions.append(action)
        return action
