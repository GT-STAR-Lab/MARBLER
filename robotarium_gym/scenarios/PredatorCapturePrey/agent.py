from robotarium_gym.utilities.misc import is_close
import numpy as np

class Agent:
    '''
    This is a helper class for pcpAgents
    Keeps track of information for each agent and creates functions needed by each agent
    This could optionally all be done in pcpAgents
    '''

    def __init__(self, index, sensing_radius, capture_radius, action_id_to_word, action_word_to_id):
        self.index = index
        self.sensing_radius = sensing_radius
        self.capture_radius = capture_radius
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
        

    def get_observation( self, state_space, agents):
        '''
            For each agent's observation-
                Checks for all prey in the range of the current agent
                Returns the closest prey if multiple agents in range
            Returns: [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
            array of dimension [1, OBS_DIM] 
        '''
        # distance from the closest prey in range
        closest_prey = -1
        # Iterate over all prey
        for p in state_space['prey']:
            # For each prey check if they are in range and get the distance
            in_range, dist = is_close(state_space['poses'], self.index, p, self.sensing_radius)
            # If the prey is in range, check if it is the closest till now
            if in_range and (dist < closest_prey or closest_prey == -1):
                prey_loc = p.reshape((1,2))[0]
                closest_prey = dist
        
        # if no prey found in range
        if closest_prey == -1:
            prey_loc = [-5,-5]
        
        observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc, self.sensing_radius, self.capture_radius])
        return observation
    
    def generate_goal(self, goal_pose, action, args):
        
        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - args.MIN_DIST, args.LEFT)
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + args.MIN_DIST, args.RIGHT)
        elif self.action_id2w[action] == 'up':
                goal_pose[1] = max( goal_pose[1] - args.MIN_DIST, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[1] = min( goal_pose[1] + args.MIN_DIST, args.DOWN)
        
        return goal_pose
                
