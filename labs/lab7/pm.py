# SYSTEM IMPORTS
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp


np.random.seed(12345)


# PYTHON PROJECT IMPORTS
from load import load


"""
    Given a transition matrix P, and convergence criteria (max number of iters and epsilon)
    This function should return the stationary distribution of P. The transition matrix input to this method must
    be connected and aperiocic.
"""
def vanilla_pm(P: np.ndarray,                       # note this is always a dense matrix
               epsilon: float=1e-9,
               max_iters=1e6
               ) -> np.ndarray:
    x: np.ndarray = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(int(max_iters)):
        x_new: np.ndarray = x @ P
        if np.linalg.norm(x_new - x, 2) < epsilon:
            return x_new
        x = x_new
    return x


"""
    Given a transition matrix P, alpha, and convergence criteria (max number of iters and epsilon)
    This function should return the stationary distribution of P. The transition matrix input to this method
    does not have to be connected and aperiodic because this implements the clever power method.
"""
def clever_pm(P: Union[np.ndarray, sp.csr_matrix],  # note the input could be dense OR sparse
              alpha: float,
              epsilon: float=1e-9,
              max_iters=1e6
              ) -> np.ndarray:
    x: np.ndarray = np.ones(P.shape[0]) / P.shape[0]
    for _ in range(int(max_iters)):
        x_new: np.ndarray = alpha * (x @ P)
        beta = np.sum(x) - np.sum(x_new)
        x_new += beta / P.shape[0]
        if np.linalg.norm(x_new - x, 2) < epsilon:
            return x_new
        x = x_new
    return x


def main() -> None:

    A: sp.coo_matrix = load()
    N: int = A.shape[0]
    vertex_incoming_degrees: np.ndarray = np.asarray(A.sum(axis=0)).reshape(-1)
    vertex_order: np.ndarray = np.argsort(vertex_incoming_degrees)

    P: np.ndarray = (A.todense() / (A.sum(axis=1) + 1e-12)).A
    for i,alpha in enumerate([0.2,0.4,0.6,0.8]):
        P_prime: np.matrix = alpha*P + (1-alpha)*1.0/N
        pi: np.ndarray = vanilla_pm(P_prime).reshape(-1)

        plt.subplot(2,2,i+1)
        plt.scatter(vertex_incoming_degrees[vertex_order], pi[vertex_order])
        plt.xlabel("incoming_deg(vertex)")
        plt.ylabel("pagerank(vertex)")
        plt.title("alpha={0:.2f}".format(alpha))

    plt.show()



if __name__ == "__main__":
    main()

