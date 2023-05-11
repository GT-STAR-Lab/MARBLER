from rps.utilities.misc import *
import matplotlib.patches as patches

from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self,args):
        self.args = args
        self.agent_marker_size_m = .15
        self.zone1_marker_size_m = args.zone1_radius
        self.line_width = 3
        self.CM = plt.cm.get_cmap('Spectral', 5) # Agent/goal color scheme
        self.show_figure = True

    def initialize_markers(self, robotarium, agents):
        agent_marker_size = determine_marker_size(robotarium, self.agent_marker_size_m)
        marker_size_zone1 = determine_marker_size(robotarium,self.zone1_marker_size_m)          

        self.goals = []
        w = self.args.end_goal_width
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([-1.5,-1], w,2, color=self.CM(4), zorder=-1)))
        self.goals.append(robotarium.axes.add_patch(patches.Rectangle([1.5-w,-1], w,2, color=self.CM(2), zorder=-1)))

        self.robot_markers = [ robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii],
                s=agent_marker_size, marker='o', facecolors='none',\
                edgecolors = (self.CM(0) if ii<agents.args.n_fast_agents else self.CM(1)), linewidth=self.line_width )\
                for ii in range(agents.num_robots) ]

        self.zone1 = robotarium.axes.scatter( \
                0, 0, s=marker_size_zone1, marker='o', facecolors='none', 
                edgecolors=self.CM(3), linewidth=self.line_width, zorder=-2) 
        self.zone1_text = robotarium.axes.text(0,0, agents.zone1_load,\
                                               verticalalignment='center', horizontalalignment='center')
        self.zone2_text = robotarium.axes.text(1.5 - w/2,0, agents.zone2_load,\
                                               verticalalignment='center', horizontalalignment='center')
    

    def update_markers(self, robotarium, agents):
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed.
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])
        self.zone1.set_sizes([determine_marker_size(robotarium, self.zone1_marker_size_m)])
        self.zone1_text.set_text(agents.zone1_load)
        self.zone2_text.set_text(agents.zone2_load)
