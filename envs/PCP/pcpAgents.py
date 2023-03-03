import numpy as np
from enum import Enum
from rps.utilities.graph import *
from utilities import *
import argparse

from PCP import pcpEnv

TYPE = Enum('TYPE', ['Predator', 'Capture'])
class Agent:
    def __init__(self, index, agent_type, sensing_radius, reward):
        self.index = index
        self.type = agent_type
        self.sensing_radius = sensing_radius
        self.reward = reward
        self.prey_loc = [] # agent hasn't found prey, nor has been communicated the location of the prey
        self.prey_found = False
        self.prey_in_range = False
        self.prey_caught = False #Always false for predator agents

    #TODO: is this really what we want for the observations?
    def get_observation( self, nbr_indices, state_space, agents):
        '''
            each agent's observation-
                poses of all neighbour agents
                checks whether the prey is within sensing radius of the agent
                or whether prey has been found by any of the neighbours
        '''
        observation = {}
        
        #Checks if the prey is in range of the agent
        if is_close(state_space['poses'], self.index , state_space['prey'], self.sensing_radius):
            self.prey_in_range = True
            if self.type == TYPE.Predator: #Only the predator agents can find the prey
                self.prey_loc = state_space['prey']
        else:
            self.prey_in_range = False
        
        # get the poses of all neighbours
        observation['neighbours'] = []
        for nbr_index in nbr_indices:
            observation['neighbours'].append( state_space['poses'][:, nbr_index ] )
            if self.prey_loc == []:
                # check if neighbour found the prey
                if len(agents[nbr_index].prey_loc) == 2:
                    print("Prey found by neighbour ", nbr_index, " communicated to agent ", self.index)
                    self.prey_loc = agents[nbr_index].prey_loc

        observation['agent_loc'] = state_space['poses'][:, self.index ]
        observation['prey_loc'] = self.prey_loc
        return observation
    
    def __str__(self):
        return f'Capture: {self.prey_caught}\nIndex: {self.index}'


class PCPAgents:
    def __init__(self, args, policies):
        # Settings
        self.max_episode_steps = args.max_episode_steps
        self.args = args
        self.policies = policies
        
        self.N_predator = args.predator
        self.N_capture = args.capture
        self.N = self.N_predator + self.N_capture
        self.rewards =[]
        self.rewards.append(np.zeros(self.N))

        self._initialize_agents(args)
        # Laplacian graph considering all agents communicating with each other (L = D - A)
        # TODO: Could change it to a dynamic, sparse graph
        #self.Laplacian = completeGL(self.N)
        self.env = pcpEnv.PCPEnv(self, args)

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

    def run_episode(self):
        '''
        Runs an episode of the simulation
        Episode will end based on what is returned in get_actions
        '''
        self.episode_steps = 0
        self.env.run_episode()
        print("Agent rewards for episode: ", sum(self.rewards))

    def get_actions(self, state_space):
        '''
        returns numpy array that is 2XNum_Robots
        The first row is the linear velocity of each robot in meters/second (range +- .03-.2)
        The second row is the angular velocity of each robot in radians/second
        Each column represents a different robot
        ''' 
        if self.episode_steps > self.max_episode_steps:
            return []
        
        #Check if every capture agent has already captured the prey and ends the episode if they have
        end_episode = True
        for a in self.agents:
            if a.type == TYPE.Capture and not a.prey_caught:
                end_episode = False
                break
        if end_episode:
            return []
        
        self.episode_steps+=1
        self.observations = self.get_observations(state_space)
        actions = []
        for i in range(self.N):
            actions.append(self.policies[i].getAction(self.observations[i], self.rewards[-1][i]))
        self.rewards.append(self.get_rewards(state_space, actions))
        
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
            nbr_indices = delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta)
            #nbr_indices = topological_neighbors(self.Laplacian, agent.index)
            observations[agent.index] = agent.get_observation(nbr_indices, state_space, self.agents)
        return observations     

    def get_rewards(self, state_space, actions):
        rewards = np.zeros(self.N)
        for i in range(self.N):
            if self.agents[i].type == TYPE.Capture and self.agents[i].prey_caught:
                rewards[i] = 0
            elif self.agents[i].prey_in_range:
                if self.agents[i].type == TYPE.Predator: 
                    rewards[i] = 0
                elif self.agents[i].type == TYPE.Capture and actions[i]["Capture"]:
                    rewards[i] = 0
                    self.agents[i].prey_caught = True
                else:
                    rewards[i] = self.agents[i].reward
            elif self.agents[i].type == TYPE.Capture and actions[i]["Capture"]:
                rewards[i] = self.agents[i].reward * 10 #BIG penalty for false capture. TODO: evaluate this
            else:
                rewards[i] = self.agents[i].reward

        return rewards


#Argparser specifcally designed for this environment. TODO: Is this really the best way?
def create_parser():
    parser = argparse.ArgumentParser(description='PCP Agents Parser')
    # predator arguments
    parser.add_argument('-predator', type=int, default=4)
    parser.add_argument('-predator_radius', type=float, default = .4)
    parser.add_argument('-predator_reward', type=float, default = -0.05)
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-capture_reward', type=float, default = -0.05)
    # environment
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-delta', type=float, default= .5)
    parser.add_argument('-max_episode_steps', type=int, default = 2000)

    return parser
