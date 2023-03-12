import argparse

#Argparser specifcally designed for this environment. TODO: Change this to a config?
def create_parser():
    parser = argparse.ArgumentParser(description='PCP Agents Parser')
    parser.add_argument('-no_capture_reward', type=float, default = -0.05)
    parser.add_argument('-capture_reward', type=float, default = 1)
    # predator arguments
    parser.add_argument('-predator', type=int, default=3)
    parser.add_argument('-predator_radius', type=float, default = .4)
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .2)
    # environment
    parser.add_argument('-show_figure', type=bool, default=False)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-delta', type=float, default= -1)
    parser.add_argument('-num_neighbors', type=float, default= 2, help='Number of neighbors to communicate with. Only used when delta <= 0')
    parser.add_argument('-max_episode_steps', type=int, default = 100)
    parser.add_argument('-update_frequency', type=int, default=20, help='Number of time steps to run the last velocities before getting new actions')
    parser.add_argument('-num_prey', type=int, default=3)

    return parser