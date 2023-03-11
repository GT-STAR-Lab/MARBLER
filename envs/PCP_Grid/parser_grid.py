import argparse

#Argparser specifcally designed for this environment. TODO: Change this to a config?
def create_parser():
    parser = argparse.ArgumentParser(description='Grid PCP Agents Parser')
    parser.add_argument('-no_capture_reward', type=float, default = -0.05)
    parser.add_argument('-capture_reward', type=float, default = 1)
    # predator arguments
    parser.add_argument('-predator', type=int, default=2)
    parser.add_argument('-predator_radius', type=float, default = 2, help='neighboring grids that the predators can see, 1 means only its own grid')
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = 1, help='Radius that the capture agents can obtain the prey. 1 means only its own grid')
    # environment
    parser.add_argument('-show_figure', type=bool, default=False)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-delta', type=float, default= 0, help="Agents will communicate with neighbors in this radius. If <=0 then will use num_neighbors")
    parser.add_argument('-num_neighbors', type=float, default= 2, help='Number of neighbors to communicate with. Only used when delta <= 0')
    parser.add_argument('-max_episode_steps', type=int, default = 30)
    parser.add_argument('-grid_size', type=int, default=3, help='Horizontal boxes of the grid is 3x this argument, verticle boxes is 2x this argument')
    parser.add_argument('-update_frequency', type=int, default=60)
    parser.add_argument('-num_prey', type=int, default=2)

    return parser