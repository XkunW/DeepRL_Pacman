# Used code from
# DQN code implemented by
# https://github.com/tychovdo/PacmanDQN

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import time
import sys

# Pacman game
from pacman import Directions
from game import Agent
from pacman_util import *

# Replay memory
from collections import deque

# Neural nets
import torch.optim as optim
from DQN import *
import datetime
import csv

params = {
    # Model backups
    'load_file': None,
    'save_file': None,
    'save_interval': 10000,

    # Training parameters
    'train_start': 1000,  # Episodes before training starts
    'batch_size': 32,  # Replay memory batch size
    'mem_size': 10000,  # Replay memory size

    'discount': 0.95,  # Discount rate (gamma value)
    'lr': 0.00025,  # Learning rate

    'num_of_actions': 4,

    # Epsilon value (epsilon-greedy)
    'epsilon': 1.0,  # Epsilon start value
    'epsilon_final': 0.1,  # Epsilon end value
    'epsilon_step': 80000,  # Epsilon steps between start and end (linear)

    # Whether a trained model is being loaded to test its performance on new data
    'model_trained_complete': True
}


class PacmanDQN(Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']

        if self.params['model_trained_complete']:
            print("Model has been trained")
            self.params['epsilon'] = 0
            if self.params['width'] == 7: #small grid layout
                print("loading the policy and target networks for smallGrid")
                self.policy_net = torch.load('pacman_policy_smallGrid_network.pt').double()
                self.target_net = torch.load('pacman_target_smallGrid_network.pt').double()
            else:
                print("loading the policy and target networks for mediumGrid")
                self.policy_net = torch.load('pacman_policy_smallClassic_network.pt').double()
                self.target_net = torch.load('pacman_policy_smallClassic_network.pt').double()
        else:
            print("Training model")
            self.policy_net = DQN_torch(self.params).double()
            self.target_net = DQN_torch(self.params).double()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.policy_net.params['lr'], alpha=0.95, eps=0.01)

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

        self.episode_r = 0
        self.delay = 0
        self.frame = 0

        #record performance at the end of each episode for plotting results
        self.win_history = [] #win rate per 100 episode
        self.reward_history = [] #average game score per 100 episode

        date_string = datetime.datetime.now().strftime("%m-%d-%H-%M")
        csvFile = open("DQN Train Performance_"  + date_string + '.csv', 'a', newline='')
        self.writer = csv.writer(csvFile)
        #add column names
        self.writer.writerow(['Episode Interval', 'Episode End','Average Reward (Score)', 'Average Win Rate'])

        # Stats
        if self.params['load_file'] is None:
            self.step = 0
        else:
            self.step = 0  # Change this later

        self.local_step = 0

        self.num_eps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

    def getMove(self):
        # Exploit / Explore
        if np.random.rand() > self.params['epsilon']:
            # Exploit action
            curr_state = torch.from_numpy(np.reshape(self.current_state,
                                                     (1, 6, self.params['height'], self.params['width'])))

            self.Q_pred = self.policy_net(curr_state.type(torch.DoubleTensor)).detach().numpy()
            self.Q_global.append(np.max(self.Q_pred))
            best_a = np.argmax(self.Q_pred)

            move = get_direction(best_a)
        else:
            # Random:
            move = get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = get_value(move)

        return move

    def observationFunction(self, state, terminal=False):
        # Do observation
        self.terminal = terminal
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = getStateMatrices(state, self.params['width'], self.params['height'])

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.  # Eat a ghost
            elif reward > 0:
                self.last_reward = 10.  # Eat a food pellet
            elif reward < -10:
                self.last_reward = -500.  # Eaten by ghost
                self.won = False
            elif reward < 0:
                self.last_reward = -1.  # Regular step

            if self.terminal and self.won:
                self.last_reward = 100.
            self.episode_r += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if params['save_file']:
                if self.local_step > self.params['train_start'] and self.local_step % self.params['save_interval'] == 0:
                    self.policy_net.save_ckpt(
                        'saves/model-' + params['save_file'] + "_" + str(self.step) + '_' + str(self.num_eps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_step += 1
        self.frame += 1
        if not self.params['model_trained_complete']:
            self.params['epsilon'] = max(self.params['epsilon_final'],
                                         1.0 - float(self.step) / float(self.params['epsilon_step']))

        return state

    def final(self, state):
        # Next
        self.episode_r += self.last_reward

        # Do observation
        self.observationFunction(state, terminal=True)

        #update target network to policy network every 100 episodes
        if self.num_eps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        #for plotting purposes, record game reward and win status
        self.reward_history.append(self.episode_r)
        self.win_history.append(self.won)

        #save the average training data to an excel file
        if self.num_eps % 100 == 0:
            #calculate statistics
            win_rate = np.sum(self.win_history)/100 #the number of games won out of 100
            avg_score = np.nanmean(self.reward_history) # the mean ignoring nans

            interval = str(self.num_eps-100) + "-" + str(self.num_eps)
            #['Episode Intervalm', 'episode end','Average Reward (Score)', 'Average Win Rate']
            self.writer.writerow([interval, self.num_eps, avg_score, win_rate])

            #reset the win rate and avg score recorders for the next 100 episodes
            self.reward_history = []
            self.win_history = []

            # save model. The file names are different depending on the layout
            if self.params['width'] == 7: #small grid layout
                if not self.params['model_trained_complete']: #no need to save model during testing
                    torch.save(self.policy_net, 'pacman_policy_smallGrid_network.pt')
                    torch.save(self.target_net, 'pacman_target_smallGrid_network.pt')
            else:
                if not self.params['model_trained_complete']: #no need to save model during testing
                    torch.save(self.policy_net, 'pacman_policy_smallClassic_network.pt')
                    torch.save(self.target_net, 'pacman_policy_smallClassic_network.pt')

        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
                         (self.num_eps, self.local_step, self.step, time.time() - self.s, self.episode_r, self.params['epsilon']))
        sys.stdout.write("| Q: %10f | won: %r \n" % (max(self.Q_global, default=float('nan')), self.won))
        sys.stdout.flush()


    def train(self):
        # Train
        if self.local_step > self.params['train_start']:
            self.step += 1
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s, batch_r, batch_a, batch_n, batch_t = zip(*batch)

            # convert from numpy to torch
            batch_s = torch.from_numpy(np.stack(batch_s)).type(torch.DoubleTensor)
            batch_r = torch.DoubleTensor(batch_r).unsqueeze(1)
            batch_a = torch.LongTensor(batch_a).unsqueeze(1)
            batch_n = torch.from_numpy(np.stack(batch_n)).type(torch.DoubleTensor)
            batch_t = 1 - np.array(batch_t)
            batch_t = torch.from_numpy(batch_t).type(torch.DoubleTensor).unsqueeze(1)

            # get Q(s, a)
            y_curr = self.policy_net(batch_s).gather(1, batch_a)

            # get Q(s', a')
            y_next = self.target_net(batch_n).detach().max(1)[0].unsqueeze(1)

            # get expected Q' values
            yj = (y_next * self.params['discount']) * batch_t + batch_r

            loss = self.criterion(y_curr, yj)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def registerInitialState(self, state):  # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_r = 0

        # Reset state
        self.last_state = None
        self.current_state = getStateMatrices(state, self.params['width'], self.params['height'])

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.num_eps += 1

    def getAction(self, state):
        move = self.getMove()

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move
