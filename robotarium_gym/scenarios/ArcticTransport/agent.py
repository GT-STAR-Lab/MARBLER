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

    def get_observation(self, env):
        '''
        The agent's observation is the following:
        [agent_x, agent_y, agent_pixel, a2_x, a2_y, a2_pixel, a3_x, a3_y, a3_pixel, a4_x, a4_y, a4_pixel,
            goal_x, goal_y, the 8 pixels surrounding drone 1, the 8 pixels surrounding drone 2]
        Where agent is this specific agent and
        if this agent is not a drone:
            agent 2 is the other agent that isn't a drone,
            agent 3 is drone 1
            agent 4 is drone 2
        if this agent is a drone:
            agent 2 is the other drone,
            agent 3 is the ice robot
            agent 4 is the water robot
        '''
        poses = env.agent_poses
        cells = []
        for i in range(env.num_robots):
            cells.append(env.get_cell_from_pose(env.agent_poses[:2, i]))
        goal_loc = env.get_pose_from_cell(env.goal_loc)

        #Appends the locations of the agents and their pixel type
        observation = np.array(poses[:, self.index][:2])
        self.pixel_type = env.grid[cells[self.index][0], cells[self.index][1]]
        if self.pixel_type == 3:
            self.reached_goal = True

        observation = np.append(observation, self.pixel_type)
        if self.index == 0:
            observation = np.append(observation, poses[:, 1][:2])
            observation = np.append(observation, env.grid[cells[1][0], cells[1][1]])
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, env.grid[cells[2][0], cells[2][1]])
            observation = np.append(observation, poses[:, 3][:2])
            observation = np.append(observation, env.grid[cells[3][0], cells[3][1]])
        elif self.index == 1:
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, env.grid[cells[0][0], cells[0][1]])
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, env.grid[cells[2][0], cells[2][1]])
            observation = np.append(observation, poses[:, 3][:2])
            observation = np.append(observation, env.grid[cells[3][0], cells[3][1]])
        elif self.index == 2:
            observation = np.append(observation, poses[:, 3][:2])
            observation = np.append(observation, env.grid[cells[3][0], cells[3][1]])
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, env.grid[cells[0][0], cells[0][1]])
            observation = np.append(observation, poses[:, 1][:2])
            observation = np.append(observation, env.grid[cells[1][0], cells[1][1]])
        elif self.index == 3:
            observation = np.append(observation, poses[:, 2][:2])
            observation = np.append(observation, env.grid[cells[2][0], cells[2][1]])
            observation = np.append(observation, poses[:, 0][:2])
            observation = np.append(observation, env.grid[cells[0][0], cells[0][1]])
            observation = np.append(observation, poses[:, 1][:2])
            observation = np.append(observation, env.grid[cells[1][0], cells[1][1]])

        observation = np.append(observation, goal_loc) #Appends the goal        

        for i in range(2):   #Appends the cells surrounding the drones
            left = cells[i][1] - 1 if cells[i][1] > 0 else cells[i][1]
            right = cells[i][1] + 1 if cells[i][1] < 11 else cells[i][1]
            up = cells[i][0] - 1 if cells[i][0] > 0 else cells[i][0]
            down = cells[i][0] + 1 if cells[i][0] < 7 else cells[i][0]
            observation = np.append(observation,env.grid[up, left])
            observation = np.append(observation,env.grid[cells[i][0], left])    
            observation = np.append(observation,env.grid[down, left])    
            observation = np.append(observation,env.grid[up, cells[i][1]])
            observation = np.append(observation,env.grid[down, cells[i][1]])
            observation = np.append(observation,env.grid[up, right])
            observation = np.append(observation,env.grid[cells[i][0], right])    
            observation = np.append(observation,env.grid[down, right])

        return observation

    def generate_goal(self, goal_pose, action, args):    
        '''
        updates the goal_pose based on the agent's:
            actions, type, and pixel it is on
        '''
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