from table import Table
from environment import Environment
from algorithm import Algorithm
import random


# on-policy TD(0) control
class Sarsa(Algorithm):
    def __init__(self, env: Environment, step_size: float, exploration_rate: float, episodes: int,
                 steps_on_each_episode: int, discounting_factor: float):
        self.env = env
        self.step_size = step_size
        self.exploration_rate = exploration_rate
        self.episodes = episodes
        self.steps_on_each_episode = steps_on_each_episode
        self.discounting_factor = discounting_factor
        self.q = Table()

    def execute(self, debug=False) -> (list, Table):
        env = self.env
        q = self.q

        episode = 0
        steps_on_each_episode = []
        while episode < self.episodes:
            episode += 1
            if debug and episode % 100 == 0:
                print("episode no. {}".format(episode))

            state = random.choice(env.get_initial_states())
            action = self._epsilon_greedy_policy(state)
            step = 0
            while step < self.steps_on_each_episode:
                step += 1
                next_state, reward = env.take_action(state, action)
                next_action = self._epsilon_greedy_policy(next_state)

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

        return steps_on_each_episode, q

    def _epsilon_greedy_policy(self, state: object):
        actions = self.env.get_actions(state)
        if random.random() <= self.exploration_rate:
            return random.choice(actions)

        best_action = None
        max_value = float('-inf')
        for action in actions:
            value = self.q.get(state, action, 0)
            if value > max_value:
                max_value = value
                best_action = action

        return best_action
