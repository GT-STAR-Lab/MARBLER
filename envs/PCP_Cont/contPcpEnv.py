import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import random
import pcpAgents

class PCPEnv:
    def __init__(self, pcpAgents, args):
        '''
        Inputs: object of the PCPAgents class and an argparse object
        Argparse object must at least contain:
            Number of predator robots as args.predator
            Number of capture robots as args.capture
            Radii for the predator and capture agents as args.predator_radius and args.capture radius respectively
            Whether or not to show the figure as args.show_figure
            Whether or not to run the simulation in real time as args.real_time
        '''
        self.args = args
        self.agents = pcpAgents
        self.num_prey = self.args.num_prey
        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey
        self.num_robots = self.args.predator + self.args.capture
        self.first_run = True 
        self.uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary()

        if self.args.show_figure:
            self.predator_marker_size_m = self.args.predator_radius
            self.capture_marker_size_m = self.args.capture_radius
            self.goal_marker_size_m = .05
            self.line_width = 5
            self.CM = plt.cm.get_cmap('hsv', 4) # Agent/goal color scheme

    def run_episode(self):
        '''
        Creates a new instance of the robotarium and runs the agents until PCPAgents.get_actions returns []
        '''
        self._create_robotarium()

        state_space, x = self._generate_state_space() #x is the poses. Can only get the poses once per step
        actions, agents = self.agents.get_actions(state_space)

        iterations = 0
        while actions != []:
            if iterations % self.args.update_frequency == 0:
                #Set the velocities to each agent based on the assigned action
                self._update_prey_status(state_space, actions, agents)
                velocities = np.array(actions).T
                velocities[0] = np.clip(velocities[0],-.25,.25)
                velocities = self.uni_barrier_cert(np.array(velocities), x) #makes sure no collisions
                self.robotarium.set_velocities(np.arange(self.num_robots), velocities)
            
            if self.args.show_figure:
                for i in range(x.shape[1]):
                    self.robot_markers[i].set_offsets(x[:2,i].T)

                    # Next two lines updates the marker sizes if the figure window size is changed. 
                    # They should be removed when submitting to the Robotarium.
                    self.robot_markers[i].set_sizes([determine_marker_size(self.robotarium, \
                                                        (self.predator_marker_size_m if i < self.args.predator else self.capture_marker_size_m))])
                #self.goal_marker.set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])
                for i in range(len(self.prey_markers)):
                    if not self.prey_captured[i]:
                        self.prey_markers[i].set_sizes([determine_marker_size(self.robotarium, self.goal_marker_size_m)])
                    else:
                        self.prey_markers[i].set_sizes([0,0])

            self.robotarium.step()
            iterations += 1
            if iterations % self.args.update_frequency == 0:
                state_space, x = self._generate_state_space()
                actions, agents = self.agents.get_actions(state_space)
            else:
                x = self.robotarium.get_poses()

    def _update_prey_status(self, state_space, actions, agents):
        removeIndicies = []
        for prey in range(len(self.prey_loc)):
            for agent in agents:
                if np.linalg.norm(state_space['poses'][:2, agent.index] - self.prey_loc[prey]) <= agent.sensing_radius:
                    self.prey_sensed[prey] = True

            if self.prey_sensed[prey]:
                for i in range(len(actions)):
                    if (actions[i]==[0,0]).all() and np.linalg.norm(state_space['poses'][:2, agents[i].index] - self.prey_loc[prey]) <= agents[i].capture_radius:
                        self.num_prey -= 1
                        self.prey_captured[prey] = True
                        removeIndicies.append(prey)
                        break
        for i in range(len(removeIndicies)-1, -1, -1):
            del self.prey_loc[removeIndicies[i]]

    def _create_robotarium(self):
        '''
        Creates a new instance of the robotarium
        Randomly initializes the prey in the right half and the agents in the left third of the Robotarium
        '''
        if self.first_run:
            self.first_run = False
        else:
            self.robotarium.call_at_scripts_end() #TODO: check if this is needed and how it affects runtime
        
        #generate initial robot locations
        #Assumes y and theta can be anything but the x locations are within the left third of the robotarium
        initial_conditions = generate_initial_conditions(self.num_robots, width=1)
        for i in range(len(initial_conditions[0])):
            initial_conditions[0][i] -= 1

        self.robotarium = robotarium.Robotarium(number_of_robots= self.num_robots, show_figure = self.args.show_figure, \
                                                initial_conditions=initial_conditions, sim_in_real_time=self.args.real_time)
        
        #setting the prey location to a random location where:
        #   x is [0,1.4]
        #   y is [-.9, .9]
        self.prey_loc = []
        for i in range(self.args.num_prey):
            prey_x = random.random() * 1.4
            prey_y = random.random() * 1.8 - .9
            self.prey_loc.append([prey_x, prey_y])

        if self.args.show_figure:
            x = self.robotarium.get_poses()

            marker_size_predator = determine_marker_size(self.robotarium, self.predator_marker_size_m)
            marker_size_capture = determine_marker_size(self.robotarium, self.capture_marker_size_m)
            marker_size_goal = determine_marker_size(self.robotarium,self.goal_marker_size_m)
            self.robot_markers = [self.robotarium.axes.scatter( \
                x[0,ii], x[1,ii], s=(marker_size_predator if ii < self.args.predator else marker_size_capture), marker='o', facecolors='none',edgecolors=self.CM(0 if ii < self.args.predator else 1),linewidth=self.line_width) 
                for ii in range(self.num_robots)]
            
            self.prey_markers = [self.robotarium.axes.scatter( \
                self.prey_loc[ii][0], self.prey_loc[ii][1], s=marker_size_goal, marker='.', facecolors=self.CM(2),edgecolors=self.CM(2),linewidth=self.line_width,zorder=-2)
                for ii in range(self.num_prey)]

        self.robotarium.step()

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        '''
        state_space = {}
        x = self.robotarium.get_poses()
        state_space['poses'] = x
        state_space['num_prey'] = self.num_prey
        state_space['prey'] = []
        for i in range(self.num_prey):
            state_space['prey'].append(np.array(self.prey_loc[i]).reshape((2,1)))

        return state_space, x


    def __del__(self):
        self.robotarium.call_at_scripts_end()

