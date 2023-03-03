import argparse

#Argparser specifcally designed for this environment. TODO: Change this to a config?
def create_parser():
    parser = argparse.ArgumentParser(description='Grid PCP Agents Parser')
    # predator arguments
    parser.add_argument('-predator', type=int, default=4)
    parser.add_argument('-predator_radius', type=float, default = 2, help='neighboring grids that the predators can see, 1 means only its own grid')
    parser.add_argument('-predator_reward', type=float, default = -0.05)
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = 1, help='Radius that the capture agents can obtain the prey. 1 means only its own grid')
    parser.add_argument('-capture_reward', type=float, default = -0.05)
    # environment
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-delta', type=float, default= .75)
    parser.add_argument('-max_episode_steps', type=int, default = 30)
    parser.add_argument('-grid_size', type=int, default=4, help='Horizontal boxes of the grid is 3x this argument, verticle boxes is 2x this argument')
    parser.add_argument('-update_frequency', type=int, default=60)

    return parser