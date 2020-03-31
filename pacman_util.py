import numpy as np
from pacman import Directions
# Used code from
# DQN code implemented by
# https://github.com/tychovdo/PacmanDQN



def getOneHot(actions, batch_size):
    """ Create list of vectors with 1 values at index of action in list """
    # actions_onehot = np.zeros((self.params['batch_size'], 4))
    actions_one_hot = np.zeros((batch_size, 4))
    for i in range(len(actions)):
        actions_one_hot[i][int(actions[i])] = 1
    return actions_one_hot


def mergeStateMatrices(state_matrices):
    """ Merge state matrices to one state tensor """
    stateMatrices = np.swapaxes(state_matrices, 0, 2)
    total = np.zeros((7, 7))
    for i in range(len(stateMatrices)):
        total += (i + 1) * stateMatrices[i] / 6
    return total


def getStateMatrices(state, width, height):
    """ Return wall, ghosts, food, capsules matrices """

    def getWallMatrix(state):
        """ Return matrix with wall coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        grid = state.data.layout.walls
        matrix = np.zeros((height, width), dtype=np.int8)
        for i in range(grid.height):
            for j in range(grid.width):
                # Put cell vertically reversed in matrix
                cell = 1 if grid[j][i] else 0
                matrix[-1 - i][j] = cell
        return matrix

    def getPacmanMatrix(state):
        """ Return matrix with pacman coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            if agentState.isPacman:
                pos = agentState.configuration.getPosition()
                cell = 1
                matrix[-1 - int(pos[1])][int(pos[0])] = cell

        return matrix

    def getGhostMatrix(state):
        """ Return matrix with ghost coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            if not agentState.isPacman:
                if not agentState.scaredTimer > 0:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell

        return matrix

    def getScaredGhostMatrix(state):
        """ Return matrix with ghost coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        matrix = np.zeros((height, width), dtype=np.int8)

        for agentState in state.data.agentStates:
            if not agentState.isPacman:
                if agentState.scaredTimer > 0:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell

        return matrix

    def getFoodMatrix(state):
        """ Return matrix with food coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        grid = state.data.food
        matrix = np.zeros((height, width), dtype=np.int8)

        for i in range(grid.height):
            for j in range(grid.width):
                # Put cell vertically reversed in matrix
                cell = 1 if grid[j][i] else 0
                matrix[-1 - i][j] = cell

        return matrix

    def getCapsulesMatrix(state):
        """ Return matrix with capsule coordinates set to 1 """
        width, height = state.data.layout.width, state.data.layout.height
        capsules = state.data.layout.capsules
        matrix = np.zeros((height, width), dtype=np.int8)

        for i in capsules:
            # Insert capsule cells vertically reversed into matrix
            matrix[-1 - i[1], i[0]] = 1

        return matrix

    # Create observation matrix as a combination of
    # wall, pacman, ghost, food and capsule matrices
    # width, height = state.data.layout.width, state.data.layout.height
    observation = np.zeros((6, height, width))

    observation[0] = getWallMatrix(state)
    observation[1] = getPacmanMatrix(state)
    observation[2] = getGhostMatrix(state)
    observation[3] = getScaredGhostMatrix(state)
    observation[4] = getFoodMatrix(state)
    observation[5] = getCapsulesMatrix(state)

    # observation = np.swapaxes(observation, 0, 2)

    return observation


def get_value(direction):
    if direction == Directions.NORTH:
        return 0.
    elif direction == Directions.EAST:
        return 1.
    elif direction == Directions.SOUTH:
        return 2.
    else:
        return 3.


def get_direction(value):
    if value == 0.:
        return Directions.NORTH
    elif value == 1.:
        return Directions.EAST
    elif value == 2.:
        return Directions.SOUTH
    else:
        return Directions.WEST
