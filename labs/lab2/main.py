# SYSTEM IMPORTS
from abc import abstractmethod, ABC     # you need these to make an abstract class in Python
from typing import List, Type, Union    # Python's typing syntax
import numpy as np                      # linear algebra & useful containers
import math                             

# PYTHON PROJECT IMPORTS


# Types defined in this module
DistributionType: Type = Type["Distribution"]
BinomialDistributionType = Type["BinomialDistribution"]
PoissonDistributionType = Type["PoissonDistribution"]
GaussianDistributionType = Type["GaussianDistribution"]


# an abstract class for an arbitrary distribution
# please don't touch this class
class Distribution(ABC):

    # this is how you make an abstract method in Python
    # all child classes MUST implement this method
    # otherwise we will get an error when the child class
    # is instantiated 
    @abstractmethod 
    def fit(self: DistributionType,
            X: np.ndarray               # input data to fit from
            ) -> DistributionType:
        ... # same as "pass"

    # another abstract method that every child class will have to implement
    # in order to be able to be instantiated
    @abstractmethod
    def prob(self: DistributionType,
             X: np.ndarray
             ) -> np.ndarray:           # return Pr[x] for each point (row) of the data (X)
        ... # same as "pass"

    @abstractmethod
    def parameters(self: DistributionType) -> List[Union[float, np.ndarray]]:
        ... # same as "pass"


# a class for the binomial distribution
# you will need to complete this class
class BinomialDistribution(Distribution):
    def __init__(self: BinomialDistributionType,
                 n: int) -> None:
        # controlled by parameter "p"
        self.p: float = None
        self.n: int = n

    def fit(self: BinomialDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> BinomialDistributionType:
        if X.shape[-1] != self.n:
            raise ValueError(f"ERROR: expected {n} outcomes in data")
        if X.shape[0] != 1:
            raise ValueError(f"ERROR: expected a single row to fit from")
        # TODO: complete me!
        y = 0
        for i in range(self.n):
            y += X[0, i]
        self.p = y / self.n
        print(f"FITTING BINOMIAL: {X=}, {self.p=}")
        return self # keep this at the end

    def prob(self: BinomialDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        if X.shape[-1] != self.n:
            raise ValueError(f"ERROR: expected 2d data with {n} outcomes in each example")
        # TODO: complete me!
        prob = np.ones((X.shape[0], 1))
        for i in range(X.shape[0]):
            for j in range(self.n):
                if X[i, j] == 1:
                    prob[i, 0] *= self.p
                else:
                    prob[i, 0] *= (1 - self.p)
        print(f"PREDICTING BINOMIAL: {X=}, {prob=}")
        return prob

    def parameters(self: BinomialDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.n, self.p]


# EXTRA CREDIT
class GaussianDistribution(Distribution):
    def __init__(self: GaussianDistributionType) -> None:
        # controlled by parameters mu and var
        self.mu: float = None
        self.var: float = None

    def fit(self: GaussianDistributionType,
            X: np.ndarray                   # input data to fit from
                                            # this will be a bunch of integer samples stored in a column vector
            ) -> GaussianDistributionType:

        # TODO: complete me!
        n = len(X)
        mu = 0
        for i in range(n):
            mu += X[i]
        self.mu = mu / n
        var = 0
        for i in range(n):
            var += (X[i] - self.mu)**2
        self.var = var / n
        print(f"FITTING GAUSSIAN: {X=}")
        return self # keep this at the end

    def prob(self: GaussianDistributionType,
             X: np.ndarray                  # this will be a column vector where every element is a float
             ) -> np.ndarray:
        # TODO: complete me!
        
        print(f"PREDICTING GAUSSIAN: {X=}")
        prob = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            x = X[i, 0]
            prob[i, 0] = (1.0 / np.sqrt(2 * np.pi * self.var)) * np.exp(-((x - self.mu)**2) / (2 * self.var))
        return prob

    def parameters(self: GaussianDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.mu, self.var]



# a class for the poisson distribution
# you will need to complete this class
class PoissonDistribution(Distribution):
    def __init__(self: PoissonDistributionType) -> None:
        # controlled by parameter "lambda"
        self.lam: float = None

    def fit(self: PoissonDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> PoissonDistributionType:
        # TODO: complete me!
        n = len(X)
        lam = 0
        for i in range(n):
            lam += X[i, 0]
        self.lam = lam / n
        print(f"FITTING POISSON: {X=}, {self.lam=}")
        return self # keep this at the end

    def prob(self: PoissonDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        # TODO: complete me!
        prob = np.ones((X.shape[0], 1))
        for i in range(X.shape[0]):
            x = X[i, 0]
            prob[i, 0] = (np.exp(-self.lam) * self.lam**x) / math.factorial(x)
        print(f"PREDICTING POISSON: {X=}, {prob=}")
        return prob

    def parameters(self: PoissonDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.lam]

