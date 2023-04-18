import rps.robotarium as robotarium
import numpy as np
import random
import time

from visualize import *
from controller import *
from pcpAgents import *
from utilities import *

class roboEnv:
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents

        self.controller = Controller()
        self.first_run = True 
        self.episodes = 0

        # Figure showing and visualizing
        self.visualizer = Visualize( self.args )
        if self.args.show_figure_frequency == -1:
            self.visualizer.show_figure = False

    def reset(self):
        '''
        Reset the environment
        '''
        self.episodes += 1
        self._create_robotarium()

    def step(self, actions_):
        '''
        Take a step into the environment given the action
        '''
        goals_ = self.agents._generate_step_goal_positions(actions_)

        # Considering one step to be equivalent to update_frequency iterations
        for iterations in range(self.args.update_frequency):
            # Get the actual position of the agents
            self.agents.agent_poses = self.robotarium.get_poses()
            # Uses the robotarium commands to get the velocities of each robot   
            # Only does this once every 10 steps because otherwise training is really slow 
            if iterations % 10 == 0 or self.args.robotarium:                    
                velocities = self.controller.set_velocities(self.agents.agent_poses, goals_)
                self.robotarium.set_velocities(np.arange(self.agents.num_robots), velocities)
            
            if self.visualizer.show_figure:
                self.visualizer.update_markers(self.robotarium, self.agents)

            self.robotarium.step()
        
        self.agents._update_tracking_and_locations(actions_)
        return self.agents._generate_state_space()
    
    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Randomly initializes the prey in the right half and the agents in the left third of the Robotarium
        '''
        # Initialize agents and tracking variables
        self.agents._initialize_tracking_and_locations()

        if self.first_run:
            self.first_run = False
        else:
            del self.robotarium

        self.robotarium = robotarium.Robotarium(number_of_robots= self.agents.num_robots, show_figure = self.visualizer.show_figure, \
                                                initial_conditions=self.agents.agent_poses, sim_in_real_time=self.args.real_time)
        self.agents.agent_poses = self.robotarium.get_poses()    
        self.robotarium.step()

        if self.visualizer.show_figure:
            self.visualizer.initialize_markers(self.robotarium, self.agents)

    def __del__(self):
        self.robotarium.call_at_scripts_end()

