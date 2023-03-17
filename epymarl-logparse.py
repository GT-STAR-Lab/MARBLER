import argparse

import re
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_logs(args,
              metrics_to_plot={
                  'test_ep_length_means': 'Mean Test Episode Length',
                  'test_return_means': 'Mean Test Episode Return',
              },
             ):
    episode_regex = r'Episode:\s+(\d+)'

    # Regular expressions to match episode number and metrics
    episode_regex = r'Episode:\s+(\d+)'
    metrics_regex = r'agent_grad_norm:\s+([\d\.]+)\s+critic_grad_norm:\s+([\d\.]+)\s+' + \
                    r'critic_loss:\s+([\d\.]+)\s+ep_length_mean:\s+([\d\.]+)\s+' + \
                    r'pg_loss:\s+([\d\.-]+)\s+q_taken_mean:\s+([\d\.-]+)\s+' + \
                    r'return_mean:\s+([\d\.-]+)\s+return_std:\s+([\d\.]+)\s+' + \
                    r'target_mean:\s+([\d\.-]+)\s+td_error_abs:\s+([\d\.]+)\s+' + \
                    r'test_ep_length_mean:\s+([\d\.]+)\s+test_return_mean:\s+([\d\.-]+)\s+' + \
                    r'test_return_std:\s+([\d\.]+)'

    # Initialize dictionary to store data
    episodes = []
    metrics = {
        'agent_grad_norms': [],
        'critic_grad_norms': [],
        'critic_losses': [],
        'ep_length_means': [],
        'pg_losses': [],
        'q_taken_means': [],
        'return_means': [],
        'return_stds': [],
        'target_means': [],
        'td_error_abs': [],
        'test_ep_length_means': [],
        'test_return_means': [],
        'test_return_stds': []
    }

    # Open the log file and iterate over each line
    with open(args.log_file, 'r') as file:
        for line in file:
            # Check if the line contains episode information
            if 'Recent Stats | t_env' in line:
                # Extract the episode number
                episode = int(re.search(episode_regex, line).group(1))
                
                if episode == 1: # episode 1 for some reason only has 2 lines instead of 4
                    continue
                episodes.append(episode)

                # Extract the corresponding metrics
                metrics_line = ''
                
                # for _ in range(4):
                #     metrics_line += file.readline()
                    
                i = 0
                while i < 4:
                    line = file.readline()
                    if 'DEBUG' not in line and 'matplotlib' not in line:
                        metrics_line += line
                        i+=1

                metrics_values = re.search(metrics_regex, metrics_line)
                for i, key in enumerate(metrics.keys()):
                    metrics[key].append(float(metrics_values.group(i+1)))

    # Plot the data for each metric against episode number
    for metric_name, metric_values in metrics.items():
        if metric_name in metrics_to_plot:
            plt.figure()
            plt.plot(episodes, metric_values)
            plt.ylabel(metrics_to_plot[metric_name])
            plt.xlabel('Episodes')
            if args.savefig:
                plt.savefig(os.path.join(os.path.dirname(args.log_file), f'{metric_name}.png'))
            if not args.noshow:
                plt.show()
            
    if args.savedf:
        df = pd.DataFrame(metrics, index=episodes)
        df.index.name = 'Episode'
        df.to_csv(os.path.join(os.path.dirname(args.log_file), f'stats.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process log file')
    parser.add_argument('--log_file', '-f', type=str, help='Path to the log file')
    parser.add_argument('--savefig', action='store_true', help='Save figures to file')
    parser.add_argument('--savedf', action='store_true', help='Save stats DataFrame to file')
    parser.add_argument('--noshow', action='store_true', help='Suppress showing of plots')

    args = parser.parse_args()
    plot_logs(args)