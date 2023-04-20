```
├── main.py           # user calls this file, specify scenario -> python -m robotarium.main --scenario pcp. Primarily to debug scenarios before training with them
├── robotarium_env    # user doesn't have to touch this folder
│   ├── controller.py
│   ├── rnn_agent.py
│   ├── roboEnv.py
│   └── utilities.py
└── scenarios
    └── pcp          # scenario user creates 
        ├── README.md
        ├── __init__.py
        ├── config.yaml  # modifies this one
        ├── model        # to store models trained for this scenario
        │   ├── agent.tiff
        │   ├── maddpg.mat
        │   ├── maddpg.tiff
        │   ├── model_config.mat
        │   ├── qmix.mat
        │   └── qmix.tiff
        ├── pcpAgents.py  # 1. main file user codes up
        ├── visualize.py  # 2. user defined visualization [TODO: Can be made more flexible - Reza]
        └── wrapper.py    # wrapper for gym [TODO: write flexible wrapper regardless of scenario - Shubham]
```

Example config [TODO: Remove redundancies?]
```
#Arguments needed by main.py
scenario: pcp #name of the folder inside scenarios

model_config_file: qmix.mat
model_file: qmix.tiff

actor_file: rnn_agent
actor_class: RNNAgent

env_file: pcpAgents
env_class: pcpAgents #This needs to have all of the functionalities of a gym to work
n_actions: 5 #The number of actions available for the agent
episodes: 10 #Number of episodes to run for

#Arguments needed by the environment
LEFT: -1.5
RIGHT: 1.5
UP: -1
DOWN: 1
ROBOT_INIT_RIGHT_THRESH : -0.5
PREY_INIT_LEFT_THRESH : 0.5
MIN_DIST : 0.1 #This should probably be renamed
time_penalty: -0.05
sense_reward: 1 
capture_reward: 5
predator: 2
predator_radius: .45
capture: 2
capture_radius: .25
show_figure_frequency: 1 #Set to -1 to turn off figures. Needs to be 1 when submitting to Robotarium
real_time: False
delta: 0.0 #THIS DOES NOT CURRENTLY WORK IF > 0
num_neighbors: 2
max_episode_steps: 80
update_frequency: 29
num_prey: 3
robotarium: True #Should be False during training to speed up robots, needs to be true when submitting
```