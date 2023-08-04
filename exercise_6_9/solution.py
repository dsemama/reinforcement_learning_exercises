from typing import Dict, List, Any, Set, Union, Tuple

import numpy
import random
from matplotlib import pyplot as plt
from environments.windy_gridworld import WindyGridworld
from policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from table import Table
from algorithms.sarsa import Sarsa
from environment import Environment
import pandas as pd

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows

Q_TABLE_FILE_NAME = 'exercise_6_9/q_table.csv'


def learn(env: Environment, q_table_filename: str = Q_TABLE_FILE_NAME):
    q_table = Table()
    algo = Sarsa(
        step_size=0.5,
        episodes=100000,
        steps_on_each_episode=100,
        discounting_factor=1,
        env=env,
        q_table=q_table,
        policy=EpsilonGreedyPolicy(env=env, q_table=q_table, exploration_rate=0.1)
    )

    steps = algo.execute(debug=True)
    plt.plot(range(len(steps)), steps)
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Reward convergence')
    plt.show(block=True)

    pd.DataFrame(_as_data(q_table)).to_csv(q_table_filename, index=False)


def play(env: Environment, q_table_filename: str = Q_TABLE_FILE_NAME) -> Set[Tuple[int, int]]:
    df = pd.read_csv(q_table_filename)
    steps = set()
    state = random.choice(env.get_initial_states())

    T = 0
    while T < 30:
        T += 1
        steps.add(state)
        if state in env.get_goal_states():
            break

        actions, values = _get_action_values(df, state)

        i = numpy.argmax(values)
        picked_action = actions[i]

        state, _ = env.take_action(state, picked_action)
    return steps


def visualize(env: Environment, steps: Set[Tuple[int, int]]):
    board = env.get_board()
    for i in range(len(board)):
        print('\n\t', end='  ')
        for j in range(len(board[0])):
            if (i, j) in env.get_goal_states():
                print('G', end='  ')
            elif (i, j) in env.get_initial_states():
                print('S', end='  ')
            elif (i, j) in steps:
                print('x', end='  ')
            else:
                print('-', end='  ')


def _get_action_values(df, state):
    actions_df = df[
        (df['position_h'] == state[0]) &
        (df['position_v'] == state[1])]
    actions_h = actions_df['action_h'].values
    actions_v = actions_df['action_v'].values
    actions = list(zip(actions_h, actions_v))
    values = actions_df['value'].values
    return actions, values


def _as_data(q: Table) -> Dict[str, List[Any]]:
    pos_h = []
    pos_v = []
    action_v = []
    action_h = []
    values = []

    for state, action in q.get_keys():
        pos_h.append(state[0])
        pos_v.append(state[1])
        action_h.append(action[0])
        action_v.append(action[1])
        values.append(q.get(state, action))

    return {
        'position_h': pos_h,
        'position_v': pos_v,
        'action_h': action_h,
        'action_v': action_v,
        'value': values
    }


def solution():
    env = WindyGridworld()

    # learn(env)

    steps = play(env)
    visualize(env, steps)
