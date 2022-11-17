#################### DO NOT CHANGE THIS FILE ####################
import optimization as hw2
import numpy as np
import optparse,sys

def gradeP1():
    score = 4
    result = hw2.sudokuCSPToGrid(hw2.sudokuCSP([(0,0,4),(0,1,3),(1,0,1),(1,2,3),(2,2,2),(3,0,2),(3,1,1)],2),2)
    if result is None:
        print("None returned - your constraints probably have no solution")
        return 0
    result = np.reshape(result,16)
    answer = [4, 3, 1, 2, 1, 2, 3, 4, 3, 4, 2, 1, 2, 1, 4, 3]
    warned = False
    for i in range(16):
        if result[i] != answer[i]:
            score = 0
            if i in {0, 1, 4, 6, 10,12,13} and not warned:
                warned = True
                print("Your code changed one of the supplied values - did you change something you shouldn't?")

    if score == 0:
        print("Your Answer:\n" + str(np.reshape(result,(4,4))))
        print("Correct Answer:\n" + str(np.reshape(answer,(4,4))))

    result = hw2.sudokuCSPToGrid(hw2.sudokuCSP(
        [(0, 0, 8), (0, 3, 9), (0, 4, 3), (0, 8, 2),
         (1, 2, 9), (1, 7, 4),
         (2, 0, 7), (2, 2, 2), (2, 3, 1), (2, 6, 9), (2, 7, 6),
         (3, 0, 2), (3, 7, 9),
         (4, 1, 6), (4, 7, 7),
         (5, 1, 7), (5, 5, 6), (5, 8, 5),
         (6, 1, 2), (6, 2, 7), (6, 5, 8), (6, 6, 4), (6, 8, 6),
         (7, 1, 3), (7, 6, 5),
         (8, 0, 5), (8, 4, 6), (8, 5, 2), (8, 8, 8)],
        3), 3)
    if result is None:
        print("None returned - your constraints probably have no solution")
        return 0

    result = np.reshape(result, 81)
    answer = [8, 4, 6, 9, 3, 7, 1, 5, 2, 3, 1, 9, 6, 2, 5, 8, 4, 7, 7, 5, 2, 1, 8, 4, 9, 6, 3, 2, 8, 5, 7, 1, 3, 6, 9,
              4, 4, 6, 3, 8, 5, 9, 2, 7, 1, 9, 7, 1, 2, 4, 6, 3, 8, 5, 1, 2, 7, 5, 9, 8, 4, 3, 6, 6, 3, 8, 4, 7, 1, 5,
              2, 9, 5, 9, 4, 3, 6, 2, 7, 1, 8]
    bonus = 1
    for i in range(81):
        if result[i] != answer[i]:
            bonus = 0

    if bonus == 0:
        print("Your Answer:\n" + str(np.reshape(result, (9, 9))))
        print("Correct Answer:\n" + str(np.reshape(answer, (9, 9))))

    print("#######################################")
    print("P1 SCORE: " + str(score+bonus) + " / 5 ")
    print("#######################################")

    return score+bonus
    
##### P4

def gradeP4():
    score = 5
    result = hw2.annealTSP(['Houston','New York City', 'Los Angeles', 'Chicago'])
    print("\n")
    if result[1] > 93.0 or result[1] < 92.0 or result[0] == ['Houston','New York City', 'Los Angeles', 'Chicago']:
        score = 0            

    if score == 0:
        print("Your Distance: " + str(result[1]))
        print("Correct Distance: 92.5725570752") 

    print("#######################################")
    print("P4 SCORE: " + str(score) + " / 5 ")
    print("#######################################")

    return score

##### P5

def gradeP5():
    correct_runs = 10
    for i in range(10):
        bad_run = False
        warned = False
        result = np.reshape(hw2.annealSudoku([(0,0,4),(0,1,3),(1,0,1),(1,2,3),(2,2,2),(3,0,2),(3,1,1)],2)[0],16)
        print("\n")
        answer = [4, 3, 1, 2, 1, 2, 3, 4, 3, 4, 2, 1, 2, 1, 4, 3]
        for i in range(16):
            if result[i] != answer[i] and bad_run == False:
                correct_runs = correct_runs - 1
                bad_run = True
        for i in range(16):
            if result[i] != answer[i] and warned == False:
                if i in {0, 1, 4, 6, 10,12,13}:
                    warned = True
                    print("Your answer changed one of the supplied values - make sure you a constraints to enforce these!")                
        if bad_run:    
            print("Your Answer:\n" + str(np.reshape(result,(4,4))))
            print("Correct Answer:\n" + str(np.reshape(answer,(4,4))))

    if correct_runs > 8:
        score = 5
    elif correct_runs > 6:
        score = 4
    elif correct_runs > 4:
        score = 3
    elif correct_runs > 2:
        score = 2
    elif correct_runs > 0:
        score = 1        
    else:
        score = 0

    print("#######################################")
    print("P5 SCORE: " + str(score) + " / 5 ")
    print("Based on: " + str(correct_runs) + " / 10 correct runs")
    print("#######################################")

    return score

##### P2

def gradeP2():
    score = 5

    result = hw2.fractionalKnapsack(5)
    if abs(result - 4.4) > 0.01:
        score = score - 1
        print("Your Answer:    " + str(result))
        print("Correct Answer: 4.4")

    result = hw2.fractionalKnapsack(7)
    if abs(result - 5.2) > 0.01:
        score = score - 1
        print("Your Answer:    " + str(result))
        print("Correct Answer: 5.2")

    result = hw2.fractionalKnapsack(10)
    if abs(result - 6.0) > 0.01:
        score = score - 1
        print("Your Answer:    " + str(result))
        print("Correct Answer: 6.0")

    result = hw2.fractionalKnapsack(2)
    if abs(result - 2.0) > 0.01:
        score = score - 1
        print("Your Answer:    " + str(result))
        print("Correct Answer: 2.0")

    result = hw2.fractionalKnapsack(1)
    if abs(result - 1.0) > 0.01:
        score = score - 1
        print("Your Answer:    " + str(result))
        print("Correct Answer: 1.0")

    print("#######################################")
    print("P2 SCORE: " + str(score) + " / 5 ")
    print("#######################################")

    return score

##### P3

def gradeP3():
    score = 4
    result = hw2.sudokuIPToGrid(hw2.sudokuIP([(0,0,4),(0,1,3),(1,0,1),(1,2,3),(2,2,2),(3,0,2),(3,1,1)],2),2)
    if result is None:
        print("None returned - maybe your problem is infeasible?")
        print("Try checking prob.status to confirm")
        score = 0
    else:
        result = np.reshape(result,16)
        answer = [4, 3, 1, 2, 1, 2, 3, 4, 3, 4, 2, 1, 2, 1, 4, 3]
        warned = False
        for i in range(16):
            if result[i] != answer[i]:
                score = 0            
                if i in {0, 1, 4, 6, 10,12,13} and not warned:
                    warned = True
                    print("Your answer changed one of the supplied values - make sure you add constraints to enforce these!")


        if score == 0:
            print("Your Answer:\n" + str(np.reshape(result,(4,4))))
            print("Correct Answer:\n" + str(np.reshape(answer,(4,4))))


    bonus = 1

    result = hw2.sudokuIPToGrid(
    hw2.sudokuIP([(0, 0, 8), (0, 3, 9), (0, 4, 3), (0, 8, 2),
        (1, 2, 9), (1, 7, 4),
        (2, 0, 7), (2, 2, 2), (2, 3, 1), (2, 6, 9), (2, 7, 6),
        (3, 0, 2), (3, 7, 9),
        (4, 1, 6), (4, 7, 7),
        (5, 1, 7), (5, 5, 6), (5, 8, 5),
        (6, 1, 2), (6, 2, 7), (6, 5, 8), (6, 6, 4), (6, 8, 6),
        (7, 1, 3), (7, 6, 5),
        (8, 0, 5), (8, 4, 6), (8, 5, 2), (8, 8, 8)], 3), 3)
    if result is None:
        print("None returned - maybe your problem is infeasible?")
        bonus = 0
    else:    
        result = np.reshape(result, 81)
        answer = [8, 4, 6, 9, 3, 7, 1, 5, 2, 3, 1, 9, 6, 2, 5, 8, 4, 7, 7, 5, 2, 1, 8, 4, 9, 6, 3, 2, 8, 5, 7, 1, 3, 6,
              9,
              4, 4, 6, 3, 8, 5, 9, 2, 7, 1, 9, 7, 1, 2, 4, 6, 3, 8, 5, 1, 2, 7, 5, 9, 8, 4, 3, 6, 6, 3, 8, 4, 7, 1,
              5,
              2, 9, 5, 9, 4, 3, 6, 2, 7, 1, 8]
        for i in range(81):
            if result[i] != answer[i]:
                bonus = 0

        if bonus == 0:
            print("Your Answer:\n" + str(np.reshape(result, (9, 9))))
            print("Correct Answer:\n" + str(np.reshape(answer, (9, 9))))



    print("#######################################")
    print("P3 SCORE: " + str(score + bonus) + " / 5 ")
    print("#######################################")

    return score+bonus



def gradeAll():
    total_points = 0
    total_points = total_points + gradeP1()
    total_points = total_points + gradeP2()
    total_points = total_points + gradeP3()
    total_points = total_points + gradeP4()
    total_points = total_points + gradeP5()
    print("#######################################")
    print("TOTAL SCORE: " + str(total_points) + " / 25 ")
    print("#######################################")

# register arguments and set default values
def readCommand(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.add_option('--question', '-q',
                    dest = 'gradeQuestion',
                    default = None,
                    type = 'int',
                    help = 'Grade one particular question.')
    (options, args) = parser.parse_args(argv)
    return options

if __name__ == '__main__':
    options = readCommand(sys.argv)
    if options.gradeQuestion == 1:
        gradeP1()
    elif options.gradeQuestion == 2:
        gradeP2()
    elif options.gradeQuestion == 3:
        gradeP3()
    elif options.gradeQuestion == 4:
        gradeP4()
    elif options.gradeQuestion == 5:
        gradeP5()
    #elif options.gradeQuestion == 6:
    #    gradeP6()
    else:
        gradeAll()
