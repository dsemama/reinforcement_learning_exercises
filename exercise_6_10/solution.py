from exercise_6_9.solution import learn, play, visualize
from environments.windy_gridworld import WindyGridworld
from exercise_6_9.windy_gridworld_agent import WindyGridWorldAgent


def solution():
    agent = WindyGridWorldAgent(
        storage_path='exercise_6_10/q_table',
        enable_stochastic_wind=True)
    agent.play(visualize=True)
