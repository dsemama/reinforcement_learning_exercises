from abc import ABC, abstractmethod
from typing import List


class Algorithm(ABC):

    @abstractmethod
    def execute(self, debug=False) -> List[float]:
        pass
