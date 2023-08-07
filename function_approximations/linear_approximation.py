import numpy as np

from function_approximation import FunctionApproximation


class LinearApproximation(FunctionApproximation):
    def eval(self, features, params) -> float:
        return np.dot(features, params).sum()

    def derivative_eval(self, features, params, derivative_param_index) -> float:
        return params[derivative_param_index]
