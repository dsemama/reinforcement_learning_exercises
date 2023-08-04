import numpy as np
import random

from environments.racing_track import RacingTrack
from table import Table

epsilon = 0.1
env = RacingTrack()


def behavior_policy(Q, state):
    if random.random() <= epsilon:
        actions = env.get_actions(state)
        return random.choice(actions)

    return target_policy(Q, state)


def action_conditional_prob(Q, state, action):
    prob = 1 - epsilon if action == target_policy(Q, state) else 0
    actions = env.get_actions(state)
    prob += epsilon * 1 / len(actions)
    return prob


def target_policy(Q, state):
    actions = env.get_actions(state)

    values = []
    for action in actions:
        value = Q.get(state, action, random.random())
        values.append(value)

    return actions[np.argmax(values)]


def algorithm():
    # Off-policy MC control

    T = 30
    Q = Table()
    C = Table()
    discounting_factor = 1
    env = RacingTrack()
    rewards_over_episodes = []
    k = 1
    episodes = 100000
    while k <= episodes:
        k += 1
        if k % 1000 == 0:
            print("episode no. {}".format(k))

        states = [env.get_initial_states()]
        actions = []
        rewards = []

        for i in range(T + 1):
            actions.append(behavior_policy(Q, states[-1]))
            state, reward = env.take_action(states[-1], actions[-1])
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

            if action == target_policy(Q, state):
                W = W * 1 / action_conditional_prob(Q, state, action)

        rewards_over_episodes.append(G)
    return rewards_over_episodes, Q
