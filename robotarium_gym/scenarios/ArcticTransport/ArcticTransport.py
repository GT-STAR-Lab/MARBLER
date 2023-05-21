import time
from gym import spaces
import numpy as np
import copy
import math
import random
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.ArcticTransport.visualize import Visualize
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_gym.scenarios.ArcticTransport.agent import Agent

class ArcticTransport(BaseEnv):
    def __init__(self, args):
        self.args = args
        self.num_robots = self.args.n_agents
        self.agent_poses = None

        self.agent_obs_dim = 30
        self.agent_action_dim = 5

        #This isn't really needed but makes a bunch of stuff clearer
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3:'down', 4:'no_action'}
        self.action_w2id = {v:k for k,v in self.action_id2w.items()}

        self.agents = [Agent(i, self.action_id2w, self.action_w2id) for i in range(self.num_robots)]
        self.agents[2].type = 'ice'
        self.agents[3].type = 'water'

        #In this environment, agent starting poses never change
        start_x = [-.3, .3, -.9, .9]
        start_y = [-.8]*4
        start_z = [math.pi/2]*self.num_robots
        self.start_poses = np.array([start_x, start_y, start_z])

        if self.args.seed != -1:
             np.random.seed(self.args.seed)

        #Initializes the action and observation spaces
        actions = []
        observations = []
        for i in range(len(self.agents)):
            actions.append(spaces.Discrete(self.agent_action_dim))
            #each agent's observation is a tuple of size 3
            obs_dim = self.agent_obs_dim
            #the minimum observation is the left corner of the robotarium
            #the maximum is the message and the pixel type which can both go up to 3
            observations.append(spaces.Box(low=-1.5, high=3, shape=(obs_dim,), dtype=np.float32))
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        
        self.visualizer = Visualize(self.args) #needed for Robotarium renderings
        #Env impliments the robotarium backend
        #It expects to have access to agent_poses, visualizer, num_robots and _generate_step_goal_positions
        self.env = roboEnv(self, args)  

    def reset(self):
        self.episode_steps = 0
        for a in self.agents:
            a.pixel_type=0
            a.reached_goal = False
        self.messages = [0,0]

        self.agent_poses = copy.deepcopy(self.start_poses)
        
        # 0 is normal terrain
        # 1 is ice
        # 2 is water
        # 3 is the goal
        # Goal is 2x2 cells in the top two rows
        # Most of the bottom row is hardset to be normal terrain
        self.grid = np.random.randint(3, size=(8, 12))
        self.goal_loc = random.randint(1, 11)
        self.grid[0][self.goal_loc] = 3
        self.grid[0][self.goal_loc - 1] = 3
        self.grid[1][self.goal_loc] = 3
        self.grid[1][self.goal_loc - 1] = 3
        self.grid[7][1:11] = np.array([0]*10)
        self.goal_loc = [1, self.goal_loc]

        self.env.reset()

        return [[0] * self.agent_obs_dim] * self.num_robots

    def step(self, actions_):
        self.episode_steps += 1

        #Robotarium actions and updating agent_poses all happen here
        message = self.env.step(actions_)

        obs = self.get_observations()
        if message == '':
            reward = self.get_reward()       
            terminated = self.episode_steps > self.args.max_episode_steps
            if not terminated:
                terminated = True
                for a in self.agents:
                    if a.type != 'drone' and not a.reached_goal:
                        terminated = False
                        break
        else:
            #print("Ending due to", message)
            reward = -30
            terminated = True

        if terminated:
            if message == '':
                print(self.episode_steps)
            else:
                print((self.args.max_episode_steps+1), message)
        
        return obs, [reward]*self.num_robots, [terminated]*self.num_robots, {}

    def get_observations(self):
        observations = []
        for a in self.agents:
            observations.append(a.get_observation(self))
        return observations

    def get_reward(self):
        reward = 0
        for a in self.agents:
            if a.type != 'drone': #Rewards are only based on how the non-drones are doing
                if not a.reached_goal:
                    reward += self.args.not_reached_penalty
                if a.pixel_type != 3:
                    dist =  np.linalg.norm(self.agent_poses[:2, a.index] - self.get_pose_from_cell(self.goal_loc))
                    reward += self.args.dist_multiplier * dist**2
        return reward
    
    def get_pose_from_cell(self,cell):
        return [cell[1] * .25 - 1.5, (-cell[0]*.25 + .75)]

    def get_cell_from_pose(self,pose):
        cell = [-int((pose[1] - 1) / .25), int((pose[0] + 1.5) / .25)]
        cell[0] = 0 if cell[0] < 0 else 7 if cell[0] > 7 else cell[0]
        cell[1] = 0 if cell[1] < 0 else 11 if cell[1] > 11 else cell[1]
        return cell

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

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
