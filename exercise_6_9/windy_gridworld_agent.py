import random
from typing import Dict, List, Any, Tuple, Set

import numpy
import pandas as pd
from matplotlib import pyplot as plt

from table import Table
from agent import Agent
from algorithms.sarsa import Sarsa
from environments.windy_gridworld import WindyGridworld
from policies.epsilon_greedy_policy import EpsilonGreedyPolicy


class WindyGridWorldAgent(Agent):
    def __init__(self,
                 storage_path='exercise_6_9/q_table.csv',
                 exploration_rate=0.1,
                 step_size=0.5,
                 episodes=100000,
                 steps_on_each_episode=100,
                 enable_stochastic_wind=False):
        self.q_table = Table()
        self.storage_path = storage_path
        self.env = WindyGridworld(enable_stochastic_wind=enable_stochastic_wind)
        self.exploration_rate = exploration_rate
        self.step_size = step_size
        self.episodes = episodes
        self.steps_on_each_episode = steps_on_each_episode

    def learn(self):
        algo = Sarsa(
            step_size=self.step_size,
            episodes=self.episodes,
            steps_on_each_episode=self.steps_on_each_episode,
            # undiscounted
            discounting_factor=1,
            env=self.env,
            q_table=self.q_table,
            policy=EpsilonGreedyPolicy(env=self.env, q_table=self.q_table, exploration_rate=self.exploration_rate)
        )

        steps = algo.execute(debug=True)
        plt.plot(range(len(steps)), steps)
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        plt.title('Reward convergence')
        plt.show(block=True)

        pd.DataFrame(self._format_table()).to_csv(self.storage_path, index=False)

    def play(self, visualize=False):
        df = pd.read_csv(self.storage_path)
        steps = set()
        state = random.choice(self.env.get_initial_states())

        T = 0
        while T < 30:
            T += 1
            steps.add(state)
            if state in self.env.get_goal_states():
                break

            actions, values = self._get_action_values(df, state)

            i = numpy.argmax(values)
            picked_action = actions[i]

            state, _ = self.env.take_action(state, picked_action)

        if visualize:
            self._visualize(steps)

    def _visualize(self, steps: Set[Tuple[int, int]]):
        board = self.env.get_board()
        for i in range(len(board)):
            print('\n\t', end='  ')
            for j in range(len(board[0])):
                if (i, j) in self.env.get_goal_states():
                    print('G', end='  ')
                elif (i, j) in self.env.get_initial_states():
                    print('S', end='  ')
                elif (i, j) in steps:
                    print('x', end='  ')
                else:
                    print('-', end='  ')

    @staticmethod
    def _get_action_values(df, state):
        actions_df = df[
            (df['position_h'] == state[0]) &
            (df['position_v'] == state[1])]
        actions_h = actions_df['action_h'].values
        actions_v = actions_df['action_v'].values
        actions = list(zip(actions_h, actions_v))
        values = actions_df['value'].values
        return actions, values

    def _format_table(self) -> Dict[str, List[Any]]:
        pos_h = []
        pos_v = []
        action_v = []
        action_h = []
        values = []

        for state, action in self.q_table.get_keys():
            pos_h.append(state[0])
            pos_v.append(state[1])
            action_h.append(action[0])
            action_v.append(action[1])
            values.append(self.q_table.get(state, action))

        return {
            'position_h': pos_h,
            'position_v': pos_v,
            'action_h': action_h,
            'action_v': action_v,
            'value': values
        }
