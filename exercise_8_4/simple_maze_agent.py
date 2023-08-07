import random
from typing import Set, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agent import Agent
from environments.simple_maze.simple_maze import SimpleMaze
from environments.simple_maze.simple_maze_state import SimpleMazeState
from policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from algorithms.tabular_dyna_q import TabularDynaQ, DeterministicModel
from table import Table


class SimpleMazeAgent(Agent):
    def __init__(self, storage_path='exercise_8_4/q_table.csv', debug=False):
        self.env = SimpleMaze()
        self.q_table = Table(timestamp_threshold=0, bonus_factor=0.1)
        self.exploration_rate = 0.1
        self.policy = EpsilonGreedyPolicy(env=self.env,
                                          q_table=self.q_table,
                                          exploration_rate=self.exploration_rate)
        self.debug = debug
        self.storage_path = storage_path
        self.model = DeterministicModel()

    def learn(self):
        algo = TabularDynaQ(
            env=self.env,
            policy=self.policy,
            model=self.model,
            step_size=0.1,
            episodes=3000,
            planning_steps=50,
            discounting_factor=0.95,
            q_table=self.q_table)

        steps = algo.execute(debug=self.debug)
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

            i = np.argmax(values)
            picked_action = actions[i]

            state, _ = self.env.take_action(state, picked_action)

        if visualize:
            self._visualize(steps)

    def _visualize(self, steps: Set[Tuple[int, int]]):
        board = self.env.get_board()
        for i in range(len(board)):
            print('\n\t', end='  ')
            for j in range(len(board[0])):
                state = SimpleMazeState(i, j)
                if state in self.env.get_goal_states():
                    print('G', end='  ')
                elif state in self.env.get_initial_states():
                    print('S', end='  ')
                elif state in steps:
                    print('*', end='  ')
                else:
                    print('-', end='  ')

    def _format_table(self) -> Dict[str, List[Any]]:
        pos_h = []
        pos_v = []
        action_v = []
        action_h = []
        values = []

        for state, action in self.q_table.get_keys():
            encoded_state = state.encode()
            pos_h.append(encoded_state[0])
            pos_v.append(encoded_state[1])
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

    @staticmethod
    def _get_action_values(df, state):
        encoded_state = state.encode()
        actions_df = df[
            (df['position_h'] == encoded_state[0]) &
            (df['position_v'] == encoded_state[1])]
        actions_h = actions_df['action_h'].values
        actions_v = actions_df['action_v'].values
        actions = list(zip(actions_h, actions_v))
        values = actions_df['value'].values
        return actions, values
