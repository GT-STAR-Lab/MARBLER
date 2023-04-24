import numpy as np

class Agent:
    '''
    This is a helper class for Simple
    Keeps track of information for each agent and creates functions needed by each agent. 
    Same agent can sense and capture the prey.
    '''

    def __init__(self, index, sensing_radius, capture_radius, action_id_to_word, action_word_to_id):
        self.index = index
        self.sensing_radius = sensing_radius 
        self.capture_radius = capture_radius
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
        

    def get_observation( self, state_space):
        '''
        Returns: [agent_x_pos, agent_y_pos, prey_x_pose, prey_y_pose]
        array of dimension [1, OBS_DIM] 
        '''
        agent_pose = np.array(state_space['poses'][:, self.index ][:2]).reshape(1,2)
        prey_loc = state_space['prey'][0].reshape(1,2)
        observation = np.concatenate((agent_pose, prey_loc), axis = 1)
        return observation
    
    def generate_goal(self, goal_pose, action, args):
        '''
        Generates the final position for each time-step for the individual
        agent.
        '''

        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - args.MIN_DIST, args.LEFT)
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + args.MIN_DIST, args.RIGHT)
        elif self.action_id2w[action] == 'up':
                goal_pose[1] = max( goal_pose[1] - args.MIN_DIST, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[1] = min( goal_pose[1] + args.MIN_DIST, args.DOWN)
        
        return goal_pose
                
