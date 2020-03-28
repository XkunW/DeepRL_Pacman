# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent

# Replay memory
from collections import deque

# Neural nets
import torch
import torch.nn.functional as F
from DQN import *

params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval': 10000,

    # Training parameters
    'train_start': 1000,  # Episodes before training starts
    'batch_size': 32,  # Replay memory batch size
    'mem_size': 10000,  # Replay memory size

    'discount': 0.99,  # Discount rate (gamma value)
    'lr': 0.0002,  # Learning rate

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,  # Epsilon start value
    'eps_final': 0.1,  # Epsilon end value
    'eps_step': 10000  # Epsilon steps between start and end (linear)
}


class PacmanDQN(Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        self.policy_net = DQN_torch(self.params).float()
        self.target_net = DQN_torch(self.params).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.criterion = F.smooth_l1_loss
        # self.criterion = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())  # , lr=self.policy_net.params['lr'])

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0
        # Defined later
        self.Q_pred = None
        self.last_action = None
        self.last_state = None
        self.current_state = None
        self.current_score = None
        self.won = None
        self.terminal = None

        # Stats
        # self.cnt = self.policy_net.sess.run(self.policy_net.global_step)
        if self.params['load_file'] is None:
            self.cnt = 0
        else:
            self.cnt = 0  # Change this later

        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getMove(self):  # change this
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            curr_state = torch.from_numpy(np.reshape(self.current_state,
                                                     (1, 6, self.params['height'], self.params['width'])))
            self.Q_pred = self.policy_net(curr_state.float()).detach().numpy()
            # print("____{}____".format(self.Q_pred))
            self.Q_global.append(np.max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def observationFunction(self, state, terminal=False):
        # Do observation
        self.terminal = terminal
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.  # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.  # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.  # Punish time (Pff..)

            if self.terminal and self.won:
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if params['save_file']:
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.policy_net.save_ckpt(
                        'saves/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.observationFunction(state, terminal=True)
        # print(type(self.Q_global))
        # print(self.Q_global)
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps']))
        sys.stdout.write("| Q: %10f | won: %r \n" % (max(self.Q_global, default=float('nan')), self.won))
        sys.stdout.flush()

    def train(self):
        # Train
        if self.local_cnt > self.params['train_start']:
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            # print(batch_s.shape)
            # print(batch_s[0].shape)
            # print(batch_s[0])

            # batch_r = np.array([batch_r, ] * 4).transpose()
            conj_batch_t = 1 - batch_t
            # conj_batch_t = np.array([conj_batch_t, ] * 4).transpose()

            self.cnt += 1

            y_curr = self.policy_net(torch.from_numpy(batch_s).permute(0, 3, 1, 2).float()) \
                .gather(1, torch.from_numpy(batch_a).long())
            # t = y_curr.gather(1, torch.from_numpy(batch_a).long())
            y_new = self.target_net(torch.from_numpy(batch_n).permute(0, 3, 1, 2).float())

            q_t = torch.max(y_new, dim=1)[0]
            yj = torch.from_numpy(batch_r).float() + torch.from_numpy(conj_batch_t) \
                 * self.policy_net.params['discount'] * q_t

            q_pred = torch.sum(y_curr * torch.from_numpy(batch_a), dim=1)

            loss = self.criterion(y_curr, yj.unsqueeze(1))
            # print(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.cost_disp = loss.item()

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
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
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state):  # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove()

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
