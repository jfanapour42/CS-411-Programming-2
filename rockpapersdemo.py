import cvxpy as cp
import numpy as np

""" Just for fum """
""" Find an optimal rock paper scissors strategy """
""" Adapted from: https://www2.cs.duke.edu/courses/fall12/cps270/lpandgames.pdf """
""" We will see more about why this works at the end of the course """
def RPS():
    A = np.array([[1, 0, -1, 1], [1, -1, 0, 1], [1, -1, 1, 0], [0, 1, 1, 1], [0, -1, -1, -1]])
    b = np.array([0, 0, 0, 1, -1])
    c = np.array([0, 0, 0, 1])
    x = cp.Variable(4)

    result = cp.Problem(cp.Maximize(
        x @ c), [A@x <= b, x >= 0, x[1] >= 0, x[2] >= 0, x[3] >= 0, x[1] + x[2] + x[3] == 1]).solve()
    print('Expected Utility: ', x.value[0], ' (Intepret this as zero)')
    print('Optimal strategy: ', x.value[1], x.value[2], x.value[3], ' (Best to play randomly!)')


def main():

    RPS()


if __name__ == '__main__':
    main()
