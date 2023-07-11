from abc import ABC, abstractmethod
from environment import Environment


class Model(ABC):
    @abstractmethod
    def set(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def get(self, state, action) -> (object, int):
        pass

    @abstractmethod
    def get_visited_state_action_pair(self):
        pass
