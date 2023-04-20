from gym import spaces, Env
from .pcpAgents import pcpAgents
from robotarium_gym.robotarium_env.utilities import objectview
import os
import yaml
module_dir = os.path.dirname(__file__)
config_path = os.path.join(module_dir, 'config.yaml')

class Wrapper(Env):
    def __init__(self):
        """Creates a Gym Wrapper for PCPAgents

        Args:
            env (PCPAgents): A PCPAgents object to wrap in a gym env
        """
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        args = objectview(config)
        self.env = pcpAgents(args)
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.n_agents = self.env.num_robots

    def reset(self):
        # Reset the wrapped environment and return the initial observation
        observation = self.env.reset()
        return observation

    def step(self, action_n):
        # Execute the given action in the wrapped environment
        obs_n, reward_n, done_n, info_n = self.env.step(action_n)
        return tuple(obs_n), reward_n, done_n, info_n
    
    def get_action_space(self):
        return self.env.get_action_space()
    
    def get_observation_space(self):
        return self.env.get_observation_space()