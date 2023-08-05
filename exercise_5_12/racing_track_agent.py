from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from agent import Agent
from environments.racing_track import RacingTrack
from policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from policies.greedy_policy import GreedyPolicy
from algorithms.off_policy_mc_control import OffPolicyMCControl
from table import Table


class RacingTrackAgent(Agent):
    def __init__(self, exploration_rate=0.1, storage_path='exercise_5_12/q_table.csv', debug=False):
        self.env = RacingTrack()
        self.q_table = Table()
        self.exploration_rate = exploration_rate

        self.exploration_rate = exploration_rate
        self.target_policy = GreedyPolicy(env=self.env, q_table=self.q_table)
        self.behavior_policy = EpsilonGreedyPolicy(env=self.env,
                                                   q_table=self.q_table,
                                                   exploration_rate=self.exploration_rate)
        self.debug = debug

        self.storage_path = storage_path

    def learn(self):
        algo = OffPolicyMCControl(
            env=self.env,
            q_table=self.q_table,
            target_policy=self.target_policy,
            behavior_policy=self.behavior_policy,
            exploration_rate=self.exploration_rate)

        rewards = algo.execute(self.debug)

        plt.plot(range(len(rewards)), rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward convergence')
        plt.show(block=True)

        pd.DataFrame(self._format_q_table()).to_csv(self.storage_path, index=False)

    def play(self, visualize=False):
        state = self.env.get_initial_states()
        df = pd.read_csv(self.storage_path)
        T = 0
        steps = set()
        while T < 30:
            steps.add(state[0])
            if state in self.env.get_goal_states():
                break

            T += 1
            actions_df = df[
                (df['position_h'] == state[0][0]) &
                (df['position_v'] == state[0][1]) &
                (df['velocity_h'] == state[1][0]) &
                (df['velocity_v'] == state[1][1])]

            actions_h = actions_df['action_h'].values
            actions_v = actions_df['action_v'].values

            actions = list(zip(actions_h, actions_v))
            values = actions_df['value'].values

            i = np.argmax(values)
            picked_action = actions[i]

            state, _ = self.env.take_action(state, picked_action)

        if visualize:
            self._visualize(steps)

    def _visualize(self, steps):
        board = self.env.get_board()
        for i in range(len(board)):
            print('\n\t', end='  ')
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    print('-', end='  ')
                elif (i, j) in steps:
                    print('x', end='  ')
                else:
                    print(' ', end='  ')

    def _format_q_table(self):
        pos_h = []
        pos_v = []
        vel_h = []
        vel_v = []
        action_v = []
        action_h = []
        values = []

        for state, action in self.q_table.get_keys():
            pos_h.append(state[0][0])
            pos_v.append(state[0][1])
            vel_h.append(state[1][0])
            vel_v.append(state[1][1])
            action_h.append(action[0])
            action_v.append(action[1])
            values.append(self.q_table.get(state, action))

        return {
            'position_h': pos_h,
            'position_v': pos_v,
            'velocity_h': vel_h,
            'velocity_v': vel_v,
            'action_h': action_h,
            'action_v': action_v,
            'value': values
        }