import math
import random

from table import Table
from environment import Environment
from algorithm import Algorithm
from policy import Policy


# n-step temporal difference algorithm
class NStepTDForEstimation(Algorithm):
    def __init__(self, env: Environment, policy: Policy, step_size: float, episodes: int,
                 time_steps: int, discounting_factor: float, n: int):
        self.env = env
        self.policy = policy
        self.step_size = step_size
        self.episodes = episodes
        self.time_steps = time_steps
        self.n = n
        self.discounting_factor = discounting_factor
        self.q = Table()

    def execute(self, debug=False) -> (list, Table):
        env = self.env
        q = self.q
        n = self.n
        d = self.discounting_factor
        episode = 0
        G = 0

        while episode < self.episodes:
            episode += 1
            if debug and episode % 100 == 0:
                print("episode no. {}".format(episode))

            state = random.choice(env.get_initial_states())

            t = 0
            T = float('inf')
            K = 0
            rewards = []
            while K != T - 1:
                t += 1

                if t < T:
                    action = self.policy.get_action(state)
                    next_state, reward = env.take_action(state, action)
                    rewards.append(reward)

                    if next_state in env.get_goal_states():
                        T = t + 1

                K = t - n + 1
                if K >= 0:
                    for k in range(K + 1, min(K + n, T) + 1):
                        G = math.pow(d, k - K - 1) * rewards[k - K - 1]
                    if K + n < T:
                        G += math.pow(d, n) * q.get(state)
