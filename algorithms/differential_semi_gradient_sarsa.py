from typing import List

import numpy as np
import random

from algorithm import Algorithm
from function_approximation import FunctionApproximation
from environment import Environment
from policy import Policy


class DifferentialSemiGradientSarsa(Algorithm):
    def __init__(self,
                 env: Environment,
                 policy: Policy,
                 function_approximation: FunctionApproximation,
                 step_size: int,
                 average_reward_step_size: int,
                 episodes: int,
                 ):
        self.env = env
        self.policy = policy
        self.function_approximation = function_approximation
        self.step_size = step_size
        self.average_reward_step_size = average_reward_step_size
        self.episodes = episodes

    def execute(self, debug=False) -> List[float]:
        state = random.choice(self.env.get_initial_states())
        action = self.policy.get_action(state)

        steps = []
        weights = np.zeros(state.get_features_dimension())
        average_reward = 0
        for i in range(self.episodes):

            if debug and i+1 % 1001 == 0:
                print("episode {}".format(i))

            next_state, reward = self.env.take_action(state, action)
            next_action = self.policy.get_action(next_state)

            steps.append(self.function_approximation.eval(state, action, weights))

            error = reward - average_reward + self.function_approximation.eval(next_state, next_action, weights) - self.function_approximation.eval(state, action, weights)
            average_reward += self.average_reward_step_size * error
            weights += self.step_size * error * self.function_approximation.derivative_eval(state, action, weights)

            state = next_state
            action = next_action



