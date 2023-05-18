import numpy as np
from gym import spaces
import copy
import yaml
import os

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.PredatorCapturePreyGNN.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.scenarios.PredatorCapturePreyGNN.agent import Agent
from rps.utilities.graph import *

class PredatorCapturePreyGNN(BaseEnv):
    def __init__(self, args):
        # Settings
        self.args = args
        # print(f"\033[0;32m{args.__dict__}\033[00m")

        module_dir = os.path.dirname(__file__)
        with open(f'{module_dir}/predefined_agents.yaml', 'r') as stream:
            self.predefined_agents = yaml.safe_load(stream)
        np.random.seed(self.args.seed)

        self.num_robots = args.n_agents
        self.agent_poses = None # robotarium convention poses
        self.prey_loc = None
        self.num_prey = args.num_prey
        
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}

        #Initializes the agents
        num_capture_agents = np.random.randint(4) + 1
        num_predator_agents = 5 - num_capture_agents
        self.agents = []

        # Initialize predator agents
        index = 0
        if self.args.test:
            agent_type = 'test_predator'
            predator_idxs = np.random.randint(self.args.n_predator_agents, size=num_predator_agents)
        else:
            agent_type = 'predator'
            predator_idxs = np.random.randint(self.args.n_test_predator_agents, size=num_predator_agents)
        for i in predator_idxs:
            self.agents.append( Agent(index, self.predefined_agents[agent_type][i]['sensing_radius'],\
                                      0, self.action_id2w, self.action_w2id, self.args) )
            index += 1

        # Initialize capture agents
        if self.args.test:
            agent_type = 'test_capture'
            capture_idxs = np.random.randint(self.args.n_test_capture_agents, size=num_capture_agents)
        else:
            agent_type = 'capture'
            capture_idxs = np.random.randint(self.args.n_capture_agents, size=num_capture_agents)
        for i in capture_idxs:
            self.agents.append( Agent(index, 0, self.predefined_agents[agent_type][i]['capture_radius'],\
                                       self.action_id2w, self.action_w2id, self.args) )
            index += 1

        if self.args.capability_aware:
            self.agent_obs_dim = 6
        else:
            self.agent_obs_dim = 4

        #initializes the actions and observation spaces
        actions = []
        observations = []      
        for i in range(self.num_robots):
            actions.append(spaces.Discrete(5))
            #The lowest any observation will be is -5 (prey loc when can't see one), the highest is 3 (largest reasonable radius an agent will have)
            observations.append(spaces.Box(low=-5, high=3, shape=(self.agent_obs_dim,), dtype=np.float32))        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))

        self.visualizer = Visualize( self.args )
        self.env = roboEnv(self, args)
        self.adj_matrix = 1-np.identity(self.num_robots, dtype=int)
             

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:,i] = agent.generate_goal(goal[:,i], actions[i], self.args)
        return goal

    def _update_tracking_and_locations(self, agent_actions):
        # iterate over all the prey
        for i, prey_location in enumerate(self.prey_loc):
            # if the prey has already been captured, nothing to be done
            if self.prey_captured[i]:
                continue        
            #check if the prey has not been sensed
            if not self.prey_sensed[i]:
                # check if any of the agents has sensed it in the current step
                for agent in self.agents:
                    # check if any robot has it within its sensing radius
                    if np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                        self.prey_sensed[i] = True
                        break

            if self.prey_sensed[i]:
                # iterative over the agent_actions determined for each agent 
                for a, action in enumerate(agent_actions):
                    # check if any robot has no_action and has the prey within its capture radius if it is sensed already
                    if self.action_id2w[action]=='no_action'\
                        and np.linalg.norm(self.agent_poses[:2, self.agents[a].index] - prey_location) <= self.agents[a].capture_radius:
                        self.prey_captured[i] = True
                        break

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        x: Poses of all the robots
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['num_prey'] = self.num_prey - sum(self.prey_captured) # number of prey not captured
        state_space['unseen_prey'] = self.num_prey - sum(self.prey_sensed) # number of prey unseen till now 
        state_space['prey'] = []

        # return locations of all prey that are not captured  till now
        for i in range(self.num_prey):
            if not self.prey_captured[i]:
                state_space['prey'].append(np.array(self.prey_loc[i]).reshape((2,1)))
        return state_space
    
    def reset(self):
        '''
        Resets the simulation
        '''
        if self.args.resample:
            #Initializes the agents
            num_capture_agents = np.random.randint(4) + 1
            num_predator_agents = 5 - num_capture_agents
            self.agents = []
        
            # Initialize predator agents
            index = 0
            if self.args.test:
                agent_type = 'test_predator'
                predator_idxs = np.random.randint(self.args.n_predator_agents, size=num_predator_agents)
            else:
                agent_type = 'predator'
                predator_idxs = np.random.randint(self.args.n_test_predator_agents, size=num_predator_agents)
            for i in predator_idxs:
                self.agents.append( Agent(index, self.predefined_agents[agent_type][i]['sensing_radius'],\
                                        0, self.action_id2w, self.action_w2id, self.args) )
                index += 1

            # Initialize capture agents
            if self.args.test:
                agent_type = 'test_capture'
                capture_idxs = np.random.randint(self.args.n_test_capture_agents, size=num_capture_agents)
            else:
                agent_type = 'capture'
                capture_idxs = np.random.randint(self.args.n_capture_agents, size=num_capture_agents)
            for i in capture_idxs:
                self.agents.append( Agent(index, 0, self.predefined_agents[agent_type][i]['capture_radius'],\
                                        self.action_id2w, self.action_w2id, self.args) )
                index += 1

        self.episode_steps = 0
        self.prey_locs = []
        self.num_prey = self.args.num_prey      
        
        # Agent locations
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.start_dist)
        
        # Prey locations and tracking
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.prey_loc = generate_initial_locations(self.num_prey, width, height, self.args.ROBOT_INIT_RIGHT_THRESH, start_dist=self.args.step_dist, spawn_left=False)
        self.prey_loc = self.prey_loc[:2].T
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        
        self.state_space = self._generate_state_space()
        self.env.reset()
        return [[0]*self.agent_obs_dim] * self.num_robots
        
    def step(self, actions_):
        '''
        Step into the environment
        Returns observation, reward, done, info
        '''
        terminated = False
        self.episode_steps += 1

        # call the environment step function and get the updated state
        return_message = self.env.step(actions_)
        
        self._update_tracking_and_locations(actions_)
        updated_state = self._generate_state_space()
        
        # get the observation and reward from the updated state
        obs     = self.get_observations(updated_state)
        rewards = self.get_rewards(updated_state)

        # penalize for collisions, record in info
        violation_occurred = 0
        if self.args.penalize_violations:
            if self.args.end_ep_on_violation and return_message != '':
                violation_occurred += 1
                # print("violation: ", return_message)
                rewards += self.args.violation_penalty
            elif not self.args.end_ep_on_violation:
                violation_occurred = return_message
                rewards +=  np.log(return_message+1) * self.args.violation_penalty #Taking the log because this can get out of control otherwise
        
        # terminate if needed
        if self.episode_steps > self.args.max_episode_steps or \
            updated_state['num_prey'] == 0:
            terminated = True    

        info = {
                "pct_captured_prey": sum(self.prey_captured) / self.num_prey,
                "total_prey": self.num_prey,
                "num_prey_captured": sum(self.prey_captured),
                "violation_occurred": violation_occurred, # not a true count, just binary for if ANY violation occurred
                } 
        
        return obs, [rewards]*self.num_robots, [terminated]*self.num_robots, info

    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            an array with [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
        '''
        if self.prey_locs == []:
            for p in state_space['prey']:
                self.prey_locs = np.concatenate((self.prey_locs, p.reshape((1,2))[0]))
        # iterate over all agents and store the observations for each in a dictionary
        # dictionary uses agent index as key
        observations = []
        neighbors = [] #Stores the neighbors of each agent if delta > -1
        for agent in self.agents: 
            observations.append(agent.get_observation(state_space, self.agents))    
            if self.args.delta > -1:
                neighbors.append(delta_disk_neighbors(state_space['poses'],agent.index,self.args.delta))
        
        #Updates the adjacency matrix
        if self.args.delta > -1:
            self.adj_matrix = np.zeros((self.num_robots, self.num_robots))
            for agents, ns in enumerate(neighbors):
                self.adj_matrix[agents, ns] = 1
        
        return observations

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        reward += (self.state_space['unseen_prey'] - state_space['unseen_prey']) * self.args.sense_reward
        reward += (self.state_space['num_prey'] - state_space['num_prey']) * self.args.capture_reward
        reward += self.args.time_penalty
        self.state_space = state_space
        return reward
    
    def get_action_space(self):
        return self.action_space
    
    def get_observation_space(self):
        return self.observation_space
