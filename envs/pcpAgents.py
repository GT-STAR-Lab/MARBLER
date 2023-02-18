import argparse
import numpy as np

from pcpEnv import *


class PCPAgents:
    def __init__(self, args):
        self.args = args
        self.env = PCPEnv(self, args)

    def run_episode(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.env.run_episode()

    def get_actions(self, state_space):
        '''
        returns numpy array that is 2XNum_Robots
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
        Each column represents a different robot
        '''
        if self.episode_steps > self.args.max_episode_steps:
            return []
        self.episode_steps+=1
        
        observations = self.get_observations(state_space)
        rewards = self.get_rewards(state_space)
        actions = np.zeros((2, self.args.sensing+self.args.capture))
        actions[0] += .1
        return actions
    
    def get_observations(self, state_space):
        return -1

    def get_rewards(self, state_space):
        rewards = np.zeros(self.args.sensing + self.args.capture)
        #for i in range(len(rewards)):
        #    np.linalg.norm(state_space['prey'], state_space[poses[]])
        return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCPAgents tester')
    parser.add_argument('-sensing', type=int, default=2)
    parser.add_argument('-sensing_radius', type=float, default = .5)
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-max_episode_steps', type=int, default = 1000)
    args = parser.parse_args()
    
    agents = PCPAgents(args)
    agents.run_episode()
    agents.run_episode()
