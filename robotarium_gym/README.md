## Creating Scenarios
### Scenario Code Structure
#### Main Class
Should inherit from `base.BaseEnv`<br>
Must include the following methods:
* `get_action_space(self)`: Returns the action space
* `get_observation_space(self)`: Returns the observation space
* `step(self, actions_)`: Takes a step. Should also call env.step(actions_). Returns [observations, rewards, done, info]
* `reset(self)`: Resets the environment. Returns an observation
* `_generate_step_goal_position(actions)`: Returns goal positions for each robot based on actions

And the following Class Variables and Objects:
* `num_agents`: Integer, the number of agents in the scenario
* `agent_poses`: 3 x num_agents array (first row is x poses, second is y poses, third is angle)
* `visualizer`: Object of the Visualize class described in the next section
* `env`: An object of type roboEnv (can be named differently)
#### Visualize
Should inherit from `base.BaseVisualization` <br>
Must include the following methods:
* `initialize_markers`: Sets the background of the Robotarium at the start of an episode
* `update_markers`: Updates the background of the Robotarium after each step
And the following class variables:
* `show_figure`: boolean, whether or not to display anything

### Wrapping and Importing
Once the scenario is created, to use it for training you must:
1. update particles in `robotarium_gym/__init__.py`
2. update the file imports and `env_dict` in `robotarium_gym/wrapper.py`

## File Structure
```
├── main.py           # user calls this file, specify scenario -> python -m robotarium.main --scenario pcp. Primarily to debug scenarios before training with them
├── __init__.py #registers specific scenarios
├── wrapper.py #wraps each scenario as a Gym Environment
├── utilities    # user doesn't have to touch this folder. Mostly to handle Robotarium calls
│   ├── controller.py
│   ├── rnn_agent.py
│   ├── roboEnv.py
│   └── utilities.py
└── scenarios
    └── Warehouse          # scenario user creates. Warehouse is probably the best template scenario
        ├── README.md
        ├── __init__.py
        ├── config.yaml  # Environment specific config file
        ├── models       # pretrained models for this scenario
        │   ├── maddpg.mat
        │   └── maddpg.tiff
        ├── Warehouse.py  # 1. main file user codes up
        ├── visualize.py  # 2. user defined visualization
```

