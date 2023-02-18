import argparse

from pcpEnv import *


class PCPAgents:
    def __init__(self, args):
        self.args = args
        self.env = PCPEnv(self, args)

    def run_episode(self):
        self.env.run_episode()

    def get_actions(self, state_space):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCPAgents tester')
    parser.add_argument('-sensing', type=int, default=2)
    parser.add_argument('-capture', type=int, default=2)
    parser.add_argument('-show_figure', type=bool, default=True)
    parser.add_argument('-real_time', type=bool, default= False)
    args = parser.parse_args()
    
    agents = PCPAgents(args)
    agents.run_episode()
