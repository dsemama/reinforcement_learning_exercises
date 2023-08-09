from abc import ABC, abstractmethod


class Agent(ABC):

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def play(self, visualize=False):
        pass
