from typing import List

import numpy as np
import random

from algorithm import Algorithm
from function_approximation import FunctionApproximation
from policies.greedy_policy import GreedyPolicy
from environment import Environment
from policy import Policy


class DifferentialSemiGradientQLearning(Algorithm):
    def __init__(self,
                 env: Environment,
                 behavior_policy: Policy,
                 function_approximation: FunctionApproximation,
                 step_size: int,
                 average_reward_step_size: int,
                 episodes: int,
                 ):
        self.env = env
        self.behaviour_policy = behavior_policy
        self.function_approximation = function_approximation
        self.step_size = step_size
        self.average_reward_step_size = average_reward_step_size
        self.episodes = episodes

        # Q Learning is an off-policy algorithm, we always take the best action
        # regardless of the action picked using the behavior_policy
        self.greedy_policy = GreedyPolicy(self.env, self.function_approximation)

    def execute(self, debug=False) -> List[float]:
        state = random.choice(self.env.get_initial_states())
        action = self.behaviour_policy.get_action(state)
        features_set = self.env.get_features_set(state, action)

        steps = []
        weights = np.zeros(len(features_set))
        average_reward = 0
        for i in range(self.episodes):
            steps.append(self.function_approximation.eval(features_set, weights))

            if debug and i + 1 % 1001 == 0:
                print("episode {}".format(i))

            next_state, reward = self.env.take_action(state, action)
            next_action = self.behaviour_policy.get_action(next_state)

            error = self._take_error_step(
                state=state,
                fs=features_set,
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

    def _take_error_step(self, state, fs, r, avg_r, w):
        greedy_pick = self.greedy_policy.get_action(state=state, weights=w)
        nfs = self.env.get_features_set(state, greedy_pick)
        return r - avg_r + self.function_approximation.eval(nfs, w) - self.function_approximation.eval(fs, w)

    @staticmethod
    def _take_average_reward_step(avg_r, step_size, error):
        return avg_r + step_size * error

    def _take_weights_step(self, step_size, error, fs, w):
        return w + step_size * error * self.function_approximation.derivative_eval(fs, w)
