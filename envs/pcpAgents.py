import argparse
import numpy as np
from enum import Enum
from rps.utilities.graph import *
from utilities import *
from pcpEnv import *
import warnings

TYPE = Enum('TYPE', ['Predator', 'Capture'])
class Agent:
    def __init__(self, index, agent_type, sensing_radius, reward):
        self.index = index
        self.type = agent_type
        self.sensing_radius = sensing_radius
        self.reward = reward
        # agent hasn't found prey, nor has been communicated the location of the prey
        self.prey_loc = []
        self.prey_found = False

    def get_observation( self, nbr_indices, state_space, agents):
        '''
            each agent's observation-
                poses of all neighbour agents
                checks whether the prey is within sensing radius of the agent
                or whether prey has been found by any of the neighbours
        '''
        observation = {}
        # get the poses of all neighbours
        observation['neighbours'] = []
        for nbr_index in nbr_indices:
            observation['neighbours'].append( state_space['poses'][:, nbr_index ] )
            if not self.prey_found:
                # check if neighbour found the prey
                if agents[nbr_index].prey_found:
                    print("Prey found by neighbour ", nbr_index, " communicated to agent ", self.index)
                    self.prey_loc = agents[nbr_index].prey_loc
                    self.prey_found = True
        
        # if prey hasnt been found check if prey is within sensing radius of the agent
        if not self.prey_found:
            if is_close(state_space['poses'], self.index , state_space['prey'], self.sensing_radius):
                print("Prey found by ", self.index, self.type.name)
                self.prey_loc = state_space['prey']
                self.prey_found = True

        return observation


class PCPAgents:
    def __init__(self, args):

        # Settings
        self.max_episode_steps = args.max_episode_steps
        self.args = args
        
        self.N_predator = args.predator
        self.N_capture = args.capture
        self.N = self.N_predator + self.N_capture
        self.rewards = np.zeros(self.N)

        self._initialize_agents(args)
        # Laplacian graph considering all agents communicating with each other (L = D - A)
        # TODO: Could change it to a dynamic, sparse graph
        self.Laplacian = completeGL(self.N)
        self.controller = create_clf_unicycle_position_controller()
        self.env = PCPEnv(self, args)

    def _initialize_agents(self, args):
        '''
        Initializes all agents and pushes them into a list - self.agents 
        predators first and then capture agents
        '''
        self.agents = []
        for i in range(self.N_predator):
            self.agents.append( Agent(i, TYPE.Predator, args.predator_radius, args.predator_reward) )
        for i in range(self.N_capture):
            self.agents.append( Agent(i + self.N_predator, TYPE.Capture, args.capture_radius, args.capture_reward) )

    def _construct_ideal_actions(self, state_space):
        goal = np.zeros((2,self.N))
        goal += state_space['prey']
        return self.controller(state_space['poses'], goal)

    def run_episode(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.rewards = np.zeros(self.N)
        self.episode_steps = 0
        self.env.run_episode()
        print("Agent rewards for episode: ", self.rewards)

    def get_actions(self, state_space):
        '''
        returns numpy array that is 2XNum_Robots
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
        Each column represents a different robot
        ''' 
        if self.episode_steps > self.max_episode_steps:
            return []
        self.episode_steps+=1
        
        observations = self.get_observations(state_space)
        self.rewards += self.get_rewards(state_space)

        actions = self._construct_ideal_actions(state_space)
        for i, agent in enumerate(self.agents):
            if agent.prey_loc == []:     
                actions[:,i] = get_random_vel()
        return actions
    
    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            a dictionary of observations for each agent with agent index as key
        '''
        # get pose and velocity of all neighbours based on laplacian graph
        observations = {}
        for agent in self.agents:
            nbr_indices = topological_neighbors(self.Laplacian, agent.index)
            observations[agent.index] = agent.get_observation(nbr_indices, state_space, self.agents)
        return observations     

    def get_rewards(self, state_space):
        rewards = np.zeros(self.N)
        for i in range(self.N):
            if is_close(state_space['poses'],i , state_space['prey'], self.agents[i].sensing_radius ):
                rewards[i] = 0
            else:
                rewards[i] = self.agents[i].reward

        return rewards

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser(description='PCPAgents tester')
    # predator arguments
    parser.add_argument('-predator', type=int, default=2)
    parser.add_argument('-predator_radius', type=float, default = .3)
    parser.add_argument('-predator_reward', type=float, default = -0.05)
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-capture_reward', type=float, default = -0.05)
    # environment
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-max_episode_steps', type=int, default = 1000)
    parser.add_argument('-goal_size', type=int, default = 0.2)
    args = parser.parse_args()
    
    agents = PCPAgents(args)
    agents.run_episode()
