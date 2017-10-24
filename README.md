# Simple Q-Learning (Reinforcement Learning)
A simple example to understand Q-Learning. The code is implemented in python and uses minimal dependencies. The GUI, Environment and the Agent are modeled in the code.

A demo video can be found here: https://www.youtube.com/watch?v=MpcAewFOFPU
![Q-Learning Demo](/simple_q_grid_demo.gif?raw=true "Demo")

## Dependencies
- Numpy
- OpenCV

## How to run
Clone the repository, move into the directory, and run the code:
```sh
$ git clone https://github.com/abhijitmajumdar/Simple_Q_Learning.git
$ cd Simple_Q_Learning
$ python simple_q_learning.py
```

## Working
When the program is run, the GUI will show the grid environment with each cell pointing to the direction that the agent is supposed to go based on the q-table values. If no values are set initially, the cells will show a circle.

The agent is represented by a orange circle. The goal is a green cell which offers a positive reward, while the red cells are terminal states with negative rewards. The black cells are obstacles where the agent cannot transition through. More such obstacles can be added or removed from the environment either at runtime by double clicking inside a cell, or in the script by defining the coordinates. The updates obstacles are updated when the GUI is refreshed.

The agent moves through the grid automatically, based on the epsilon-greedy policy set by the EPSILON value, which causes it to explore and learn. It can be observed that as the agent moves from one cell to another, the q-values are updated and the corresponding best action might change, which will be reflected by the arrow in the cell. Once the agent moves into a red or green cell the episode ends and a new one begins with the agent reseting its position to the start.

The episodes are run in background, since they are much faster, however, for every GUI_UPDATE_STEP, an episode is shown in the GUI. If any obstacle is added or removed, they are updated during this run. The episode GUI can be skipped (for example if taking too much time), by pressing 's' in the GUI window.
##### Environment
The environment is initialized using the GUI object and the grid details. It has methods take action and return state and reward based on the system dynamics, get state, reset agent state and redefine the grid.
##### Agent
The agent object is initialized grid size (to form the q-table), the hyper parameters and whether to save the learned q-table or not. It has methods to choose an action based on the epsilon-greedy approach, it can ask the environment to take a step, update its q-table based on the reward and methods for loading and saving the q-table.
##### Q-Learning
The essential component of the Q-learning algorithm is the manner in which it updates the q-table for the agent. This is show below:

!!!!!Put equation here

##### Graphical User Interface (GUI)
The GUI is initialized with the dimensions, the grid details and the option to capture mouse inputs. The initialization draws the grid world with all objects in the grid. The update() method is used to redraw the agent, the best actions to take in each cell and the episode number in the grid world environment. It also has methods to reset the frame(when new obstacles are added/removed) and check mouse input.

## Parameters
The parameters can be changed in the 'simple_q_learning.py' file based on the definitions below:
- LOAD_AND_SAVE_LEARNED_MODEL: True if you want to continue the learning from where you left off
- SAVE_FILENAME: The filename to save the learned q-table
- GUI_REFRESH_RATE: Lower the value, faster the runs
- GUI_UPDATE_STEP: How frequently do you want the GUI to show an episode
- SHOW_STEP: Debug print frequency
- GRID_SIZE: Grid dimensions
- GRID_TERMINALS: Coordinates in grid with negative reward
- GRID_OBSTACLES: Coordinates in grid where agent cannot go
- GOAL: Positive reward awarded at this(these) coordinate(s)
- GUI_DIMENSION: GUI dimensions in pixels
- ACTIONS: List of actions possible
- EPSILON: Value in range->(0,1). Towards 0 -> Greedy, towards 1 -> Exploration
- GAMMA: Discount factor for rewards
- ALPHA: Learning rate
- N_EPISODES: Maximum number of episodes to run


## Changes
- Add a save learned value and load learned value
- Display episode number
- Fixed problem with terminal state update of q-table
- Add multiple obstacles in one gui update
- Remove existing obstacles by clicking on them again
- Added GUI update iteration constant, so GUI is updated only once very $constant$ times, learning is much faster in the background

## To-do
- Add an option at the beginning to select whether to run learning or run a learned run
- Add inputs to control the number of episodes(+-100 to the current)
- Add a new state when discovered, dont initiate at the beginning
- Dynamic exploration changes:
    - Keep e low when learning new grid and keep track of number of steps to goal (exploration at the beginning)
    - as we learn to reach goal(number of steps reduce), increase e (try to follow learned path)
    - if sort-of convergence observed(number of steps not reducing any more), decrease e again (see if better solution is available in the near proximity solutions)
    - if no new states are initiated OR if number of steps to goal dont still reduce, increase e again (if no better solution observed, abide to the learned solution)
- If new obstacles are introduced find a way to disregard the previously learned moves around the new obstacle
