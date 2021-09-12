# V2G-Predictor
## Diploma theses 2020-21
- **Author**: Antonios Valmas
- **University**: NTUA School of electrical and computer engineering
- **Supervisor**: Emmanouel Varvarigos
- **Description**: Optimize the V2G charging system via reinforcement learning and keeping the complexity of the problem independent of the capacity of the parking.

## Source code

### Requirements
- Python 3.8.10  (https://www.python.org/)
- pip 21.0.1 (If it isn't installed along with python, check this https://packaging.python.org/tutorials/installing-packages/)
- *Optional*: virtualenv 20.0.28 (https://pypi.org/project/virtualenv/)

### Optional: Create virtualenv
Python 3.8.10 must be already installed in your system and accessible through the command line
For example, running the following

```sh
python3.8 --version
```

The output should be:

```sh
Python 3.8.10
```

You can create a virtualenv on the root folder of your project by executing the following:
```sh
virtualenv -p python3.8 venv
```

This will create a `venv` folder containing all files of the python environment

**Important**: In order to use it you will need to run the following on every command line instance you might need it
```sh
source ./venv/bin/activate
```

### Dependencies
All dependencies of the project are documented in the `requirements.txt` in the root folder fo the project.
To install them run the following:
```sh
pip install -r requirements.txt
```

### Run
To train the policy
```sh
python3 run.py
```

To evaluate the latest trained policy
```sh
python3 eval.py
```

To export the final plots from the eval data
```
python3 plot.py
```

To test the functionalities of the environment
```sh
python3 test.py
```

## Abstract modules
The folder `app/abstract` contains two abstract modules that can be re-used
The `ddqn` module uses the `utils` module and depends on the folder structure in order to detect it.
So if you need to re-use them in a different project under a different project please edit the `line 6` on the `ddqn.py` file and correct the module path

### DDQN
The `ddqn.py` file contains the necessary code to initialize the DDQN agent and train it on a given environment and network.
A usage of the module can be found in the `app/policies/dqn.py` file

### Utils
Contains a function `compute_avg_return`, which computes the average return for a given environment and a given number of episodes to run.
Used on the validation step on the `train` function
