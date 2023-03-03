import argparse

#Argparser specifcally designed for this environment. TODO: Change this to a config?
def create_parser():
    parser = argparse.ArgumentParser(description='PCP Agents Parser')
    # predator arguments
    parser.add_argument('-predator', type=int, default=4)
    parser.add_argument('-predator_radius', type=float, default = .4)
    parser.add_argument('-predator_reward', type=float, default = -0.05)
    # capture arguments
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-capture_radius', type=float, default = .15)
    parser.add_argument('-capture_reward', type=float, default = -0.05)
    # environment
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    parser.add_argument('-delta', type=float, default= .5)
    parser.add_argument('-max_episode_steps', type=int, default = 2000)

    return parser