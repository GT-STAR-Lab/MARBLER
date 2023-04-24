from rps.utilities.misc import *
from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self, args):
        self.predator_marker_size_m = args.predator_radius
        self.capture_marker_size_m = args.capture_radius
        self.goal_marker_size_m = .05
        self.line_width = 1
        self.CM = plt.cm.get_cmap('hsv', 7) # Agent/goal color scheme
        self.show_figure = True
    
    def initialize_markers(self, robotarium, agents):
        '''
        Initializes the marker for 
        '''
        marker_size_agent = determine_marker_size(robotarium, self.predator_marker_size_m)
        marker_size_goal = determine_marker_size(robotarium,self.goal_marker_size_m)          

        self.robot_markers = [ robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii], 
                s=marker_size_agent, \
                marker='o', facecolors='none',\
                edgecolors = self.CM(0), linewidth=self.line_width )\
                for ii in range(agents.num_robots) ]
        
        self.prey_markers = [robotarium.axes.scatter( \
                agents.prey_loc[ii][0], agents.prey_loc[ii][1], \
                s=marker_size_goal, marker='.', facecolors=self.CM(2), 
                edgecolors=self.CM(2), linewidth=self.line_width, zorder=-2) for ii in range(agents.num_prey)]
    
    def update_markers(self, robotarium, agents ):

        # TODO: May need to fix the color after agent is sensed
        
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed. 
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, \
                (self.predator_marker_size_m))])
        
        # update prey marker color if sensed, remove if captured
        for i in range(agents.num_prey):
            if not agents.prey_captured[i]:
                self.prey_markers[i].set_sizes([determine_marker_size(robotarium, self.goal_marker_size_m)])
                # change color if sensed
                if agents.prey_sensed[i]:
                    self.prey_markers[i].set_facecolor(self.CM(4))
            else:
                self.prey_markers[i].set_sizes([0,0])