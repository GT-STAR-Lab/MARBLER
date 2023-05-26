from rps.utilities.misc import *
import matplotlib.patches as patches
from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self,args):
        self.args = args
        self.agent_marker_size_m = .15
        self.line_width = 3
        self.show_figure = True
        self.CM = plt.cm.get_cmap('cool', 16) # Agent/goal color scheme
    
    def initialize_markers(self, robotarium, agents):
        agent_marker_size = determine_marker_size(robotarium, self.agent_marker_size_m)
        self.robot_markers = [ robotarium.axes.scatter( \
                agents.agent_poses[0,ii], agents.agent_poses[1,ii],
                s=agent_marker_size, marker='o', facecolors='none',\
                edgecolors = (self.CM(10) if ii<2 else self.CM(2) if ii == 2 else self.CM(5)),\
                linewidth=self.line_width ) for ii in range(agents.num_robots) ]
        
        for i,x in enumerate(agents.grid):
            for j,y in enumerate(x):
                col = self.CM(0) if y == 1 else self.CM(7) if y == 2 else self.CM(12)
                loc = agents.get_pose_from_cell([i,j])
                robotarium.axes.add_patch(patches.Rectangle(loc, .25, .25, color = col, zorder = -1))

    def update_markers(self, robotarium, agents):
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed.
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])