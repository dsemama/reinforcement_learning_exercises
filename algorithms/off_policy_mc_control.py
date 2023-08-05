import random
from typing import List

from algorithm import Algorithm
from environment import Environment
from policy import Policy
from table import Table


# racing track env
class OffPolicyMCControl(Algorithm):
    def __init__(self,
                 env: Environment,
                 q_table: Table,
                 target_policy: Policy,
                 behavior_policy: Policy,
                 exploration_rate: int):
        self.env = env
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy,
        self.exploration_rate = exploration_rate
        self.q_table = q_table

    def execute(self, debug=False) -> List[float]:
        T = 30
        Q = self.q_table
        C = Table()
        discounting_factor = 1
        rewards_over_episodes = []
        k = 1
        episodes = 100000
        while k <= episodes:
            k += 1
            if debug and k % 1000 == 0:
                print("episode no. {}".format(k))

            states = [self.env.get_initial_states()]
            actions = []
            rewards = []

            for i in range(T + 1):
                actions.append(self.behavior_policy.get_action(states[-1]))
                state, reward = self.env.take_action(states[-1], actions[-1])
                rewards.append(reward)
                states.append(state)

            G = 0
            W = 1

            for i in reversed(range(T)):
                G = discounting_factor * G + rewards[i + 1]
                state = states[i]
                action = actions[i]
                C.set(state, action, C.get(state, action, 0) + W)

                if C.get(state, action, 0) != 0:
                    Q.get(state, action, random.random())
                    Q.set(state, action, Q.get(state, action) + W / C.get(state, action) * (G - Q.get(state, action)))

                if action == self.target_policy.get_action(state):
                    W = W * 1 / self.action_conditional_prob(state, action)

            rewards_over_episodes.append(G)
        return rewards_over_episodes, Q

    def action_conditional_prob(self, state, action):
        prob = 1 - self.exploration_rate if action == self.target_policy.get_action(state) else 0
        actions = self.env.get_actions(state)
        prob += self.exploration_rate * 1 / len(actions)
        return prob
