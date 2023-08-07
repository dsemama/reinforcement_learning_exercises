from abc import abstractmethod, ABC


class FunctionApproximation(ABC):

    @abstractmethod
    def eval(self, features, params) -> float:
        pass

    @abstractmethod
    def derivative_eval(self, features, params, derivative_param_index) -> float:
        pass
