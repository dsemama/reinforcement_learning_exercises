from abc import ABC, abstractmethod

from typing import List


class Environment(ABC):

    @abstractmethod
    def get_initial_states(self) -> list:
        pass

    @abstractmethod
    def get_goal_states(self) -> list:
        pass

    @abstractmethod
    def take_action(self, current_state, action) -> (object, int):
        # returns (next_state, reward)
        pass

    @abstractmethod
    def get_actions(self, state) -> list:
        pass

    @abstractmethod
    def get_board(self) -> List[List[int]]:
        pass

