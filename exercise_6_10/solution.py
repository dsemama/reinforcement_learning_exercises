from exercise_6_9.solution import learn, play, visualize
from environments.windy_gridworld import WindyGridworld

Q_TABLE_FILENAME='exercise_6_10/q_table'

def solution():
    env = WindyGridworld(enable_stochastic_wind=True)
    # learn(env, q_table_filename=Q_TABLE_FILENAME)

    steps = play(env, q_table_filename=Q_TABLE_FILENAME)
    visualize(env, steps)
