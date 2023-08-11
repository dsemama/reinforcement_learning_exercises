from abc import abstractmethod, ABC
from typing import List


class FunctionApproximation(ABC):

    @abstractmethod
    def eval(self, features, params) -> float:
        pass

    @abstractmethod
    def derivative_eval(self, features, params) -> List[float]:
        pass
