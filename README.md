# Pruning-based analysis of input sensory relevance in decentralized controllers of voxel-based soft robots

This repository contains the code valid for the `Bio-Inspired Artificial Intelligence` course project of the
master Artificial Intelligence Systems at the University of Trento, Italy.

The implementation of the `hnn` module has taken inspiration from the [SBM repository](https://github.com/ndr09/sbm).

The code simulates the evolution of a population of soft robots controlled by decentralized controllers, one for each voxel.
The controllers are neural networks with a fixed topology, and the weights are updated through Hebbian learning. The
`cma-es` evolutionary algorithm is used to optimize the `ABCD` rules of the Hebbian learning.

The environment is simulated using the [`evogym` library](https://evolutiongym.github.io/).

## Getting started

### Installation

The code supports Python 3.8 and above. To install the required dependencies, first it is required to install the
`evogym` library. To do so, follow the instructions in the [official repository](https://evolutiongym.github.io/tutorials/getting-started.html#download).

Then, install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### Running the code

The code can be run using the `main.py` script in the `src` directory. The script accepts the following arguments:

| Parameter                                         | Description                                                                                                                         | Required | Default                                     |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------|---------------------------------------------|
| `--robot`                                         | The robot used. At the moment, the `qorm` is the only supported.                                                                    | `False`  | worm                                        |
| `--env`                                           | The environment of the _evogym_ library to simulate the robot. Supported values are _walking_flat_, _down_stepper_ and _soft_bridge | `False`  | walking_flat                                |
| `--network`                                       | The architecture of the networks assigned to the controllers. Supported values are _hnn_                                            | `False`  | hnn                                         |
| `----evo-algo-type`                               | The evolutionary algorithms. Supported values are _cma-es_                                                                          | `False`  | hnn                                         |
| `--nodes`                                         | The nodes of the architecture chosen. The considered inputs are 15 and the outputs are 2. It can accepts also hidden layers.        | `False`  | [15, 2]                                     |
| `--eta`                                           | The eta of the nueral networks.                                                                                                     | `False`  | 0.1                                         |
| `--robot-structure-path`                          | The path where to store/load the structure of the robot.                                                                            | `False`  | `../data/robot_structure/worm/default.json` |
| `--random-structure`                              | Whether to generate a random structure for the robot or not.                                                                        | `False`  | `False`                                     |
| `--train`                                         | Whether to train the robot.                                                                                                         | `False`  | `False`                                     |
| `--test`                                          | Whether to test the robot.                                                                                                          | `False`  | `False`                                     |
| `--prune`                                         | Whether to prune the robot.                                                                                                         | `False`  | `False`                                     |
| `--weight-path`                                   | The path to store/load the controllers _ABCD_ rules.                                                                                | `True`   |                                             |
| `--generations`                                   | The number of generations to train the robot.                                                                                       | `False`  | 30                                          |
| `--offsprings`                                    | The number of offsprings every generation defines.                                                                                  | `False`  | 15                                          |
| `--population-size`                               | The size of the population of the evolutionary algorithm.                                                                           | `False`  | 4                                           |
| `--sigma`                                         | The sigma value of the evolutionary algorithm.                                                                                      | `False`  | 4                                           |
| `--max_steps`                                     | The maximum steps the individual do in the environment.                                                                             | `False`  | 2000                                        |
| `--weight_update_steps`                           | The number of steps the individual do before updating its weights.                                                                  | `False`  | 150                                         |
| `--prune_ratio`                                   | The prune ratio to apply during the pruning phase.                                                                                  | `False`  | 60                                          |
| `--weight_pruning_time`                           | The update weights time where to apply the pruning.                                                                                 | `False`  | 5                                           |
| `--multi-processing`                              | Supported for training, it allows to run multiple individuals simulations in parallel.                                              | `False`  | `False`                                     |
| `--raise-error-in-case-of-loading-structure-path` | Whether to raise error in case the robot structure path is wrong or used the default structure.                                     | `False`  | `True`                                      |
| `--display`                                       | Whether to display the robot movements during test and pruning simulations.                                                         | `False`  | `False`                                     |
