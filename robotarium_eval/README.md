# How to Run on the Robotarium
The Robotarium only accepts files, not folders. Additionally, the file types it accepts is limited.
Therefore, the following steps are necessary. <br>

1. Copy the actor file you are using to this folder
    - The actor files are typically located in `src/modules/agents` within epymarl
    - The standard agents from epymarl are already here
2. Copy the saved weights that you wish to run on the Robotarium here and change the extension from `.th` to something allowed by the Robotarium
    - Weights are typically saved in `results/models/` in epymarl
3. Copy the corresponding config file for the weights and change the extension from `.json` to something allowed by the Robotarium
    - These are typically saved in `results/sacred/trainingName/environmentName/expNumber/config.json` in epymarl
3. Copy the all of the files needed for the environment
    - It is likely that include paths will need to be changed because there can be no subdirectories
4. Update config.npy. The first 9 lines are required for all experiments. The last part is the environment specific config file
    - This is really a yaml file
    - When submitting, show_figure_frequency needs to be 1 and robotarium should be true
5. Submit main.py, config.npy, the model config, the model weights, the actor file, and the environment files to the [Robotarium](https://www.robotarium.gatech.edu/dashboard) and set main.py as your Main file

## Note on File Types
The acceptable files types in the Robotarium are ` .m`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.tiff`, `.bmp`, `.py`, `.mat`, `.npy`, and `.mexa64 ` <br>
Therefore, I am using `.tiff` for `.th` files, `.mat` for `.json` files, and `.npy` for `.yaml` files. It is not required you follow these conventions.

## Sources
To find out more about EPyMarl and see where the actor code in this folder came from, see [Georgios Papoudakis et al. 2021](https://github.com/uoe-agents/epymarl)