# How to Run on the Robotarium
The Robotarium only accepts files, not folders. Additionally, the file types it accepts is limited.
Therefore, the `generate_submission` script was created <br>

## Usage

1. Make sure that everything is working as expected in robotarium_gym
    - Can be checked by running `python3 -m robotarium_gym.main`
2. Run `python3 generate_submission.py --scenario SCENARIO_NAME --name SUBMISSION_NAME`
    - This will generate the Robotarium submission files in robotarium_submissionSUBMISSION_NAME
3. Make sure config.npy is correct
    - Specifically, ensure that show_figure_frequency is 1 and robotarium is True
4. Submit submit everything in robotarium_submissionSUBMISSION_NAME to the [Robotarium](https://www.robotarium.gatech.edu/dashboard) and set main.py as your Main file

## Note on File Types
The acceptable files types in the Robotarium are ` .m`, `.jpg`, `.jpeg`, `.png`, `.gif`, `.tiff`, `.bmp`, `.py`, `.mat`, `.npy`, and `.mexa64 ` <br>
Therefore, I am using `.tiff` for `.th` files, `.mat` for `.json` files, and `.npy` for `.yaml` files. It is not required you follow these conventions.

## Testing
This has only been tested on agents trained using epymarl