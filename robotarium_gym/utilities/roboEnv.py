import copy
import rps.robotarium as robotarium
import numpy as np
import random
import time

#This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.utilities.controller import *
from robotarium_gym.utilities.misc import *

class roboEnv:
    def __init__(self, agents, args):
        self.args = args
        self.agents = agents
        if "barrier_certificate" in self.args.__dict__.keys():
            self.controller = Controller(self.args.barrier_certificate)
        else:
            self.controller = Controller()
        self.first_run = True 
        self.episodes = 0
        self.errors = {"collision":0, "boundary":0}

        # Figure showing and visualizing
        self.visualizer = agents.visualizer
        

    def reset(self):
        '''
        Reset the environment
        '''
        self.episodes += 1
        if self.args.show_figure_frequency == -1 or self.episodes % self.args.show_figure_frequency > 0:
            self.visualizer.show_figure = False
        else:
            self.visualizer.show_figure = True
        self._create_robotarium()

    def step(self, actions_):
        '''
        Take a step into the environment given the action
        '''
        goals_ = self.agents._generate_step_goal_positions(actions_)
        message = ''
        num_errors = 0
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

            #If checking for violations (boundary and collision) then
            #   Check if there has been a violation ever and then check if the number of times that violation has occurred has increased since the last timestep
            
            if self.args.penalize_violations:
                if 'end_ep_on_violation' in self.args.__dict__.keys() and not self.args.end_ep_on_violation:
                    if 'collision' in self.robotarium._errors:
                        if 'collision' not in self.errors:
                            num_errors += sum(self.robotarium._errors['collision'].values())
                        elif sum(self.robotarium._errors['collision'].values()) > sum(self.errors['collision'].values()):
                            num_errors += sum(self.robotarium._errors['collision'].values()) - sum(self.errors['collision'].values())
                    if 'boundary' in self.robotarium._errors: 
                        if 'boundary' not in self.errors:
                            num_errors += sum(self.robotarium._errors['boundary'].values())
                        elif sum(self.robotarium._errors['boundary'].values()) > sum(self.errors['boundary'].values()):
                            num_errors += sum(self.robotarium._errors['boundary'].values()) - sum(self.errors['boundary'].values())
                    self.errors = copy.deepcopy(self.robotarium._errors)
                else:
                    if 'collision' in self.robotarium._errors and ('collision' not in self.errors or sum(self.robotarium._errors['collision'].values()) > sum(self.errors['collision'].values())):
                        message = 'collision'  
                    if 'boundary' in self.robotarium._errors and ('boundary' not in self.errors or sum(self.robotarium._errors['boundary'].values()) > sum(self.errors['boundary'].values())):
                            if message == '':
                                message = 'boundary'
                            else:
                                message += "_boundary"
                    self.errors = copy.deepcopy(self.robotarium._errors)
                    if message != '':
                        return message
        if 'end_ep_on_violation' in self.args.__dict__.keys() and not self.args.end_ep_on_violation:
            return num_errors
        return ""
    
    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Randomly initializes the prey in the right half and the agents in the left third of the Robotarium
        '''
        # Initialize agents and tracking variables
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

