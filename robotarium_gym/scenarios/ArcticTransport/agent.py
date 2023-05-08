import numpy as np

class Agent:
    def __init__(self, index, action_id_to_word, action_word_to_id, type='drone'):
        '''
        Type can be: "ice", "drone", or "water"
        '''
        self.index = index
        self.action_id2w = action_id_to_word
        self.action_w2id = action_word_to_id
        self.type=type
        self.pixel_type = 0
        self.reached_goal = False

    def get_observation(self, grid, cell, goal_loc, poses, messages):
        '''
        Cell is the location of the grid cell the agent is currently occupying
        '''

        #Appends the locations of the agents
        observation = np.array(poses[:, self.index][:2])
        if self.index == 0:
            observation = np.append(observation, poses[:, 1][:2])
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, poses[:, 3][:2])
        elif self.index == 1:
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, poses[:, 3][:2])
        elif self.index == 2:
            observation = np.append(observation, poses[:, 3][:2])
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, poses[:, 1][:2])
        elif self.index == 3:
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, poses[:, 1][:2])

        observation = np.append(observation, messages) #Appends the messages
        observation = np.append(observation, goal_loc) #Appends the goal
        
        pixel_type = grid[cell[0], cell[1]]
        observation = np.append(observation, pixel_type)
        self.pixel_type = pixel_type
        if self.pixel_type == 3:
            self.reached_goal = True

        if self.type == 'drone':   
            left = cell[1] - 1 if cell[1] > 0 else cell[1]
            right = cell[1] + 1 if cell[1] < 11 else cell[1]
            up = cell[0] - 1 if cell[0] > 0 else cell[0]
            down = cell[0] + 1 if cell[0] < 7 else cell[0]
            observation = np.append(observation,grid[up, left])
            observation = np.append(observation,grid[cell[0], left])    
            observation = np.append(observation,grid[down, left])    
            observation = np.append(observation,grid[up, cell[1]])
            observation = np.append(observation,grid[down, cell[1]])
            observation = np.append(observation,grid[up, right])
            observation = np.append(observation,grid[cell[0], right])    
            observation = np.append(observation,grid[down, right])
        else:
            for i in range(8):
                observation = np.append(observation,-1)

        return observation

    def generate_goal(self, goal_pose, action, args):    
        '''
        updates the goal_pose based on the agent's:
            actions, type, and pixel it is on
        '''
        action = action // 5 #This is because the agent's action size is 20 to accomodate communication

        if self.type == 'drone':
            step_dist = args.fast_step
        elif self.type == 'water':
            if self.pixel_type == 0:
                step_dist = args.normal_step
            elif self.pixel_type == 1:
                step_dist = args.slow_step
            elif self.pixel_type == 2:
                step_dist = args.fast_step
            else:
                step_dist = args.normal_step
        else: #Type must be 'ice'
            if self.pixel_type == 0:
                step_dist = args.normal_step
            elif self.pixel_type == 1:
                step_dist = args.fast_step
            elif self.pixel_type == 2:
                step_dist = args.slow_step
            else:
                step_dist = args.normal_step

        if self.action_id2w[action] == 'left':
                goal_pose[0] = max( goal_pose[0] - step_dist, args.LEFT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
                goal_pose[0] = min( goal_pose[0] + step_dist, args.RIGHT)
                goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = max( goal_pose[1] - step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
                goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
                goal_pose[1] = min( goal_pose[1] + step_dist, args.DOWN)
        else:
             goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                      args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
             goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                      args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        return goal_pose