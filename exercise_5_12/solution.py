import numpy
from matplotlib import pyplot as plt
from table import Table
import pandas as pd
from environments.racing_track import RacingTrack
from algorithms.off_policy_mc_control import OffPolicyMCControl
from policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from policies.greedy_policy import GreedyPolicy

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows

Q_TABLE_FILE_NAME = 'exercise_5_12/q_table.csv'
env = RacingTrack()


def learn():
    exploration_rate = 0.1
    q_table = Table()
    algo = OffPolicyMCControl(
        env=RacingTrack(),
        q_table=q_table,
        target_policy=GreedyPolicy(env=env, q_table=q_table),
        behavior_policy=EpsilonGreedyPolicy(env=env, q_table=q_table, exploration_rate=exploration_rate),
        exploration_rate=exploration_rate)
    rewards, Q = algo.execute(True)

    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward convergence')
    plt.show(block=True)

    pd.DataFrame(as_data(Q)).to_csv(Q_TABLE_FILE_NAME, index=False)


def play():
    state = env.get_initial_states()
    df = pd.read_csv(Q_TABLE_FILE_NAME)
    T = 0
    steps = set()
    while T < 30:
        steps.add(state[0])
        if state in env.get_goal_states():
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

        i = numpy.argmax(values)
        picked_action = actions[i]

        state, _ = env.take_action(state, picked_action)
    return steps


def visualize(steps):
    board = env.get_board()
    for i in range(len(board)):
        print('\n\t', end='  ')
        for j in range(len(board[0])):
            if board[i][j] == 0:
                print('-', end='  ')
            elif (i, j) in steps:
                print('x', end='  ')
            else:
                print(' ', end='  ')


def as_data(q: Table):
    pos_h = []
    pos_v = []
    vel_h = []
    vel_v = []
    action_v = []
    action_h = []
    values = []

    for state, action in q.get_keys():
        pos_h.append(state[0][0])
        pos_v.append(state[0][1])
        vel_h.append(state[1][0])
        vel_v.append(state[1][1])
        action_h.append(action[0])
        action_v.append(action[1])
        values.append(q.get(state, action))

    return {
        'position_h': pos_h,
        'position_v': pos_v,
        'velocity_h': vel_h,
        'velocity_v': vel_v,
        'action_h': action_h,
        'action_v': action_v,
        'value': values
    }


def solution():
    steps = play()
    visualize(steps)
