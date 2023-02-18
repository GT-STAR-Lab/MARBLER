import argparse
import numpy as np

from pcpEnv import *


class PCPAgents:
    def __init__(self, args):
        self.args = args
        self.env = PCPEnv(self, args)

    def run_episode(self):
        self.env.run_episode()

    #returns numpy array that is 2XNum_Robots
    #The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
    #The second row is the angular velocity of each robot in radians/second
    def get_actions(self, state_space):
        observations = self.get_observations(state_space)
        rewards = self.get_rewards(state_space)
        actions = np.zeros((2, self.args.sensing+self.args.capture))
        actions[0] += .1
        return actions
    
    def get_observations(self, state_space):
        return -1

    def get_rewards(self, state_space):
        rewards = np.zeros(self.args.sensing + self.args.capture)
        for i in len(rewards):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCPAgents tester')
    parser.add_argument('-sensing', type=int, default=2)
    parser.add_argument('-sensing_radius', type=float, deafult = .5)
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    args = parser.parse_args()
    
    agents = PCPAgents(args)
    agents.run_episode()
