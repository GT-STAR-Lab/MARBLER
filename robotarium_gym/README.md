## How to Create a New Scenario

## File Structure
```
├── main.py           # user calls this file, specify scenario -> python -m robotarium.main --scenario pcp. Primarily to debug scenarios before training with them
├── utilities    # user doesn't have to touch this folder. Mostly to handle Robotarium calls
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
        └── wrapper.py    # wrapper for gym
```

