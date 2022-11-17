import cvxpy as cp

""" An example of how to set up and solve an integer problem """
""" Hint: Don't forget to specify integer=True in variable creation"""
def integerProgrammingExample():
    # Create two scalar optimization variables.
    x = cp.Variable(integer=True)
    y = cp.Variable(integer=True)

    # Create two constraints.
    constraints = [x + y == 10, x-y >= 1, x >= 0, y >=0]

    # Form objective.
    obj = cp.Minimize(x)

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    print(prob.solve())


def main():

    integerProgrammingExample()


if __name__ == '__main__':
    main()
