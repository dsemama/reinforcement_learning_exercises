from table import Table
from environment import Environment
from algorithm import Algorithm
from policy import Policy
import random


# on-policy TD(0) control
class Sarsa(Algorithm):
    def __init__(self, env: Environment, policy: Policy, q_table: Table, step_size: float, episodes: int,
                 steps_on_each_episode: int, discounting_factor: float):
        self.env = env
        self.step_size = step_size
        self.episodes = episodes
        self.steps_on_each_episode = steps_on_each_episode
        self.discounting_factor = discounting_factor
        self.policy = policy
        self.q_table = q_table

    def execute(self, debug=False) -> list:
        env = self.env
        q = self.q_table

        episode = 0
        steps_on_each_episode = []
        while episode < self.episodes:
            episode += 1
            if debug and episode % 100 == 0:
                print("episode no. {}".format(episode))

            state = random.choice(env.get_initial_states())
            action = self.policy.get_action(state)
            step = 0
            while step < self.steps_on_each_episode:
                step += 1
                next_state, reward = env.take_action(state, action)
                next_action = self.policy.get_action(next_state)

                s = state
                a = action
                r = reward
                d = self.discounting_factor
                ss = next_state
                aa = next_action
                step_size = self.step_size
                state_action_value = q.get(s, a, 0)
                next_state_action_value = q.get(ss, aa, 0)
                q.set(s, a, state_action_value + step_size * (r + d * next_state_action_value - state_action_value))

                if next_state in env.get_goal_states():
                    break

                state = next_state
                action = next_action
            steps_on_each_episode.append(step)

        return steps_on_each_episode
