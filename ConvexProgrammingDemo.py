import cvxpy as cp
import numpy as np

""" Very simple least-squares program example using cvxpy"""


def convexProgrammingSimple():

    # This is a trivial convex program to help with using cvxpy
    # We will use a simple least-sqaures problem to demonstrate some of the functionality.
    # These are of the form:

    # Matrices and vectors are represented as numpy arrays. There are several ways to create numpy arrays,
    # Here we will create and fill them directly.
    # First, here is a matrix (numpy array) with one value in it, the number 4
    A = np.array([4])  # Note that np.array() takes a list so the number 4 is in brackets

    # Here is a second matrix with one value in it, the number 1
    b = np.array([1])

    # Define 1 variable to solve for.
    x = cp.Variable(1)

    # We will define the cost that we wish to minimize as the squared value of Ax - b
    # Which in this case is: (4x-1)^2
    # Note the use of @ in Matrix-Vector multiplication instead of *
    cost = cp.sum_squares(A @ x - b)

    # Note that this is similar to the above example: cp.norm(x - A[ i,:] ,2)
    # Try it:
    #cost = cp.norm(A @ x - b, 2)

    # We use the squared difference here for a few reasons, but for this assignment it is
    # important to remember that the solution works with any constraints you may have set,
    # which in this example none are explicitly set. If we wanted to minimize x for Ax - b without
    # squaring it, the optimal value would be -infinity!
    # As there is no optimal solution it will report None!
    # Try for yourself and use this instead:
    #cost = A @ x - b

    # Solution:
    # Because this is set up as solving a single equation, we can do it before hand:
    # From calculus, the minimum is found when the derivative is equal to 0
    # d/dx (4x - 1)^2 = 2(4x-1) = 0
    # 8x - 2 = 0
    # 8x = 2
    # x = 1/4

    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    # Print result.
    print('The optimal value is', prob.value,  '(NOTE: this is should be interpreted as being 0)')
    print('The optimal x is')
    print(x.value)
    print('What this says is that for minimizing (4x-1)^2, the optimal value of x is 0.25, which gives the optimal value (4(0.25) - 1)^2 = 0')


def convexProgrammingConstrained():

    # another example using cvxpy with constraints.
    # Dimensions of matrix A
    n = 2
    m = 10

    # Create matrix A
    # Fills matrix with sample(s) drawn from a normal distribution. (will be 0 <= x <= 1)
    A = np.random.randn(m, n)

    # Two variables
    x = cp.Variable(n)

    # Set up what we want to minimize (the sum of squares)
    # Note that the loop is over the total rows (going down the matrix A)
    # Sidenote: think of cp.norm( ,2) as distance (Euclidean)
    #           and think of the whole process as minimizing the total sum of distances
    f = sum([cp.norm(x - A[i, :], 2) for i in range(m)])

    # We create a Problem, pass in the objective function and tell cvxpy we want to minimize it
    # And then ask it to solve.
    constraints = [sum(x) == 0]
    result = cp.Problem(cp.Minimize(f), constraints).solve()

    # We want the values of x
    print(x.value)

    # Try it: What is the optimal value when x is minimize?
    print(result)


def main():

    print('convexProgrammingSimple:')
    convexProgrammingSimple()

    print('convexProgrammingConstrained:')
    convexProgrammingConstrained()


if __name__ == '__main__':
    main()
