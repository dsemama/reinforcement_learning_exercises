from abc import ABC, abstractmethod
from typing import List


class State(ABC):

    @abstractmethod
    def encode(self) -> List[int]:
        pass

    @staticmethod
    @abstractmethod
    def decode(decoded_state: List[int]):
        pass

    @abstractmethod
    def get_feature_set(self, action) -> List[float]:
        pass

    @abstractmethod
    def get_features_dimension(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass
