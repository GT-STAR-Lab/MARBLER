from robotarium_gym.utilities.misc import is_close
import numpy as np

class Agent:
    '''
    This is a helper class for PredatorCapturePrey
    Keeps track of information for each agent and creates functions needed by each agent
    This could optionally all be done in PredatorCapturePrey
    '''

    def __init__(self, index, sensing_radius, capture_radius, action_id_to_word, action_word_to_id, capability_aware):
        self.index = index
        self.sensing_radius = sensing_radius
        self.capture_radius = capture_radius
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
        self.capability_aware = capability_aware
        

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
        
        if self.capability_aware:
            observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc, self.sensing_radius, self.capture_radius])
        else:
            observation = np.array([*state_space['poses'][:, self.index ][:2], *prey_loc])
        return observation
    
    def generate_goal(self, goal_pose, action, args):
        '''
        Sets the agent's goal to step_dist in the direction of choice
        Bounds the agent by args.LEFT, args.RIGHT, args.UP and args.DOWN
        '''

        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - args.step_dist, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + args.step_dist, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + args.step_dist, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose
                
