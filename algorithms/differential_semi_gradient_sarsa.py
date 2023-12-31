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
        features_set = self.env.get_features_set(state, action)

        steps = []
        weights = np.zeros(len(features_set))
        average_reward = 0
        for i in range(self.episodes):

            if debug and i + 1 % 1001 == 0:
                print("episode {}".format(i))

            next_state, reward = self.env.take_action(state, action)
            next_action = self.policy.get_action(next_state)
            next_feature_set = self.env.get_features_set(next_state, next_action)

            steps.append(self.function_approximation.eval(features_set, weights))

            error = self._take_error_step(
                fs=features_set,
                nfs=next_feature_set,
                r=reward,
                avg_r=average_reward,
                w=weights,
            )

            average_reward = self._take_average_reward_step(
                avg_r=average_reward,
                step_size=self.average_reward_step_size,
                error=error,
            )

            weights = self._take_weights_step(
                step_size=self.step_size,
                error=error,
                fs=features_set,
                w=weights
            )

            state = next_state
            action = next_action
            features_set = self.env.get_features_set(state, action)
        return steps

    def _take_error_step(self, fs, nfs, r, avg_r, w):
        return r - avg_r + self.function_approximation.eval(nfs, w) - self.function_approximation.eval(fs, w)

    @staticmethod
    def _take_average_reward_step(avg_r, step_size, error):
        return avg_r + step_size * error

    def _take_weights_step(self, step_size, error, fs, w):
        return w + step_size * error * self.function_approximation.derivative_eval(fs, w)
