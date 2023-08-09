from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def get_action(self, state, weights=None):
        pass
