import constraint


"""" An example of how to set up and solve a CSP for the map of Australia using constraint package"""


def graphColoringExample():

    # First step is to initialize a Problem() from the 'constraint' package.
    colorProblem = constraint.Problem()

    # Next, we will define the values the variables can have, and add them to our problem.
    domains = ["red", "green", "blue"]
    colorProblem.addVariable("WA", domains)  # Ex: "WA" can be "red", "green", or "blue"
    colorProblem.addVariable("NT", domains)
    colorProblem.addVariable("Q", domains)
    colorProblem.addVariable("NSW", domains)
    colorProblem.addVariable("V", domains)
    colorProblem.addVariable("SA", domains)
    colorProblem.addVariable("T", domains)

    # We then add in all of the constraints that must be satisfied.
    # In the map coloring problem we are doing, this means that each section
    # cannot be the same color as any of its neighbors.
    # There are other types of constraints you can add if you want to look at the documentation.
    # However, this type will suffice to do the assignment.
    # Ex: WA can't be same value (color) as NT
    colorProblem.addConstraint(lambda a, b: a != b, ("WA", "NT"))
    colorProblem.addConstraint(lambda a, b: a != b, ("WA", "SA"))
    colorProblem.addConstraint(lambda a, b: a != b, ("SA", "NT"))
    colorProblem.addConstraint(lambda a, b: a != b, ("Q", "NT"))
    colorProblem.addConstraint(lambda a, b: a != b, ("SA", "Q"))
    colorProblem.addConstraint(lambda a, b: a != b, ("NSW", "SA"))
    colorProblem.addConstraint(lambda a, b: a != b, ("NSW", "Q"))
    colorProblem.addConstraint(lambda a, b: a != b, ("SA", "V"))
    colorProblem.addConstraint(lambda a, b: a != b, ("NSW", "V"))

    # The constraint problem is now fully defined and we let the solver take over.
    # We call getSolution() and print it.
    print(colorProblem.getSolution())


def main():

    graphColoringExample()


if __name__ == '__main__':
    main()
