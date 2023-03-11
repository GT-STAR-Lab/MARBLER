from rps.utilities.controllers import *
import numpy as np

from utilities import *

def getNeighborsPrey(observations):
    for i in range(2, len(observations), 6):
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
    
    def getAction(self, observations, critic_observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        stop_dist = .3 #Ideally this is learned in a real agent
        action = 'stop'
        if (getNeighborsPrey(observations) == [-1,-1]).all():
            if observations[0] < 1.2:
                action = 'right'
            elif observations[1] > -.8:
                action = 'up'
            else:
                action = 'down'
        else:                
            if observations[1] < getNeighborsPrey(observations)[1] - stop_dist:
                action = 'down'
            elif observations[0] < getNeighborsPrey(observations)[0] - stop_dist:
                action = 'right'
            elif observations[0] > getNeighborsPrey(observations)[0] + stop_dist:
                action = 'left'
            elif observations[1] > getNeighborsPrey(observations)[1] + stop_dist:
                action = 'up'
            else:
                action = 'stop'
       
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

    def getAction(self, observations, critic_observations, prevReward):
        self.observations.append(observations)
        self.rewards.append(prevReward)
        action = 'stop'
        stop_dist = observations[5]

        if (getNeighborsPrey(observations) == [-1,-1]).all():
            if observations[0] < 1.2:
                action = 'right'
            elif observations[1] > -.8:
                action = 'up'
            else:
                action = 'down'
        else:
            if observations[1] < getNeighborsPrey(observations)[1] - stop_dist:
                action = 'down'
            elif observations[0] < getNeighborsPrey(observations)[0] - stop_dist:
                action = 'right'
            elif observations[0] > getNeighborsPrey(observations)[0] + stop_dist:
                action = 'left'
            elif observations[1] > getNeighborsPrey(observations)[1] + stop_dist:
                action = 'up'
            else:
                action= 'capture'

        self.actions.append(action)
        return action
