import torch
import json
import importlib
import numpy as np
import yaml

config_path = "grid.yaml"

class DictView(object):
    def __init__(self, d):
        self.__dict__ = d

def load_model(args):
    model_config = open(args.model_config_file)
    model_config = DictView(json.load(model_config))
    model_config.n_actions = args.n_actions

    params = torch.load(args.model_file, map_location=torch.device('cpu'))
    input_dim = params[list(params.keys())[0]].shape[1]

    actor = importlib.import_module(args.actor_file)
    actor = getattr(actor, args.actor_class)
    
    model = actor(input_dim, model_config)
    model.load_state_dict(params)

    #print(model.eval())
    return model, model_config

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

args = DictView(config)
model, model_config = load_model(args)

env_module = importlib.import_module(args.env_file)
env_class = getattr(env_module, args.env_class)
env = env_class(args)

obs = np.array(env.reset())
n_agents = len(obs)
hs = [np.zeros((model_config.hidden_dim, )) for i in range(n_agents)]

totalReward = 0
totalSteps = 0
for i in range(args.episodes):
    episodeReward = 0
    episodeSteps = 0
    for j in range(args.max_episode_steps):      
        if model_config.obs_agent_id: #Appends the agent id if obs_agent_id is true. TODO: support obs_last_action too
            obs = np.concatenate([obs,np.eye(n_agents)], axis=1)

        #Gets the q values and then the action from the q values
        q_values, hs = model(torch.Tensor(obs), torch.Tensor(hs))
        actions = np.argmax(q_values.detach().numpy(), axis=1)

        obs, reward, done, _ = env.step(actions)
        episodeReward += reward[0]
        if done[0]:
            episodeSteps = j+1
            break
    if episodeSteps == 0:
        episodeSteps = args.max_episode_steps
    obs = env.reset()
    print('Episode', i+1)
    print('Episode reward:', episodeReward)
    print('Episode steps:', episodeSteps)
    totalReward += episodeReward
    totalSteps += episodeSteps
print('\nTotal Reward:',totalReward)
print('Total Steps:', totalSteps)
