# the first number - second number = the number of test cases
# the second number = number of training episodes
python pacman.py -p PacmanDQN -n 6100 -x 6000 -l smallGrid

conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
"""
Available layouts: (2 grids for training, 2 grids for testing)
- smallGrid: for training
- smallGrid_test: for testing the model trained on smallGrid
- mediumGrid: for training
- mediumGrid_test: for testing the model trained on mediumGrid


For example, to train the model, call:
python pacman.py -p PacmanDQN -n 10100 -x 10000 -l smallGrid

To test the model on a new grid:
- first go to PacmanDQN_Agents.py, and change line 51 to: 'model_trained_complete': True
- Then run: python pacman.py -p PacmanDQN -n 100 -x 0 -l smallGrid_test

"""
