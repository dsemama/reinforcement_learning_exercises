from typing import Dict, List, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from agent import Agent

from algorithms.differential_semi_gradient_q_learning import DifferentialSemiGradientQLearning
from environments.bairds_counterexample.bairds_counterexample_environment import BairdsCounterexampleEnvironment
from environments.bairds_counterexample.policies import BehaviorPolicy, TargetPolicy
from environments.bairds_counterexample.approximation import Approximation


class BairdsCounterexampleAgent(Agent):
    def __init__(self,
                 storage_path='exercise_11_3/weights.csv',
                 discounting_factor=0.99,
                 step_size=0.9,
                 episodes=1000):
        self.env = BairdsCounterexampleEnvironment()
        self.behavior_policy = BehaviorPolicy()
        self.approximation = Approximation()
        self.episodes = episodes
        self.discounting_factor = discounting_factor
        self.storage_path = storage_path
        self.weights = None

        # stability doesn't depend on the specific step size, as long as is a positive integer.
        # larger step sizes would affect the rate at which the weights goes to infinity, but
        # not whether it goes there or not.
        self.step_size = step_size
        self.average_reward_step_size = step_size

    def learn(self):
        algo = DifferentialSemiGradientQLearning(
            env=self.env,
            behavior_policy=self.behavior_policy,
            function_approximation=self.approximation,
            step_size=self.step_size,
            average_reward_step_size=self.average_reward_step_size,
            episodes=self.episodes,
        )

        steps = algo.execute(debug=True)
        x = range(len(steps))
        w = []
        for step in steps:
            for i in range(len(step)):
                if len(w) < i+1:
                    w.append([])
                w[i].append(np.log(step[i]))

        for i in range(len(steps[0])):
            plt.plot(x, w[i], label="w{}".format(i+1))

        plt.plot(range(len(steps)), steps)
        plt.xlabel('Episodes')
        plt.ylabel('Weights')
        plt.title('Weights divergence (log(weights))')
        plt.show(block=True)

        pd.DataFrame(self._format_table(steps[-1])).to_csv(self.storage_path, index=False)

    def play(self, visualize=False):
        raise NotImplemented("In this example we demonstrate divergence due to the "
                             "usage of the deadly triage (page no. 264)")

    @staticmethod
    def _format_table(weights) -> Dict[str, List[Any]]:
        return {
            'weights': weights,
        }
