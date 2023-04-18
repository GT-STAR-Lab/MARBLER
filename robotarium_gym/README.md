## Base files that users don't change - 

`roboEnv.py`  The boss `env` class holds the complete robotarium environment, calls agents functions from agent class defined by user (described below)
`utilities.py` holds utility functions available
`controller.py` holds the robotarium dynamics controller - abstracted away to avoid confusion

## Files that user changes according to need - 
`pcpAgents.py` - an `agent` class with functionality required for the current scenario. TODO: Make it more clear what is the interface between `agents` and `env` class
`visualizer.py` - according to the required scenario

## More TODOs-
- Superclass for agents class abstracting away functions user doesn't need to implement
- Better file structure - put stuff that users don't touch inside a folder
- Make the agents code cleaner if possible
- Make documentation for the specific function definitions required by the `env` class needs in order to function
- Visualization modularization. - take as input number of agents, their positions etc  -- user doesn't need to deal with plotting bs at all

## `env` class interface functions necessary to be implemented in the `agents` class

```
_generate_step_goal_positions(actions_)
_update_tracking_and_locations(actions_)
_generate_state_space()
_initialize_tracking_and_locations()
```
