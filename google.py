#!/usr/bin/env python2.7

from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------------------------------------------------
            # LEVEL 4 (Q1)
# ------------------------------------------------------------------------------------------------------------



# Distract the Trainers
# =====================

# The time for the mass escape has come, and you need to distract the bunny trainers so that the workers can make it out! Unfortunately for you, they're watching the bunnies closely. Fortunately, this means they haven't realized yet that the space station is about to explode due to the destruction of the LAMBCHOP doomsday device. Also fortunately, all that time you spent working as first a minion and then a henchman means that you know the trainers are fond of bananas. And gambling. And thumb wrestling.

# The bunny trainers, being bored, readily accept your suggestion to play the Banana Games.

# You will set up simultaneous thumb wrestling matches. In each match, two trainers will pair off to thumb wrestle. The trainer with fewer bananas will bet all their bananas, and the other trainer will match the bet. The winner will receive all of the bet bananas. You don't pair off trainers with the same number of bananas (you will see why, shortly). You know enough trainer psychology to know that the one who has more bananas always gets over-confident and loses. Once a match begins, the pair of trainers will continue to thumb wrestle and exchange bananas, until both of them have the same number of bananas. Once that happens, both of them will lose interest and go back to supervising the bunny workers, and you don't want THAT to happen!

# For example, if the two trainers that were paired started with 3 and 5 bananas, after the first round of thumb wrestling they will have 6 and 2 (the one with 3 bananas wins and gets 3 bananas from the loser). After the second round, they will have 4 and 4 (the one with 6 bananas loses 2 bananas). At that point they stop and get back to training bunnies.

# How is all this useful to distract the bunny trainers? Notice that if the trainers had started with 1 and 4 bananas, then they keep thumb wrestling! 1, 4 -> 2, 3 -> 4, 1 -> 3, 2 -> 1, 4 and so on.

# Now your plan is clear. You must pair up the trainers in such a way that the maximum number of trainers go into an infinite thumb wrestling loop!

# Write a function solution(banana_list) which, given a list of positive integers depicting the amount of bananas the each trainer starts with, returns the fewest possible number of bunny trainers that will be left to watch the workers. Element i of the list will be the number of bananas that trainer i (counting from 0) starts with.

# The number of trainers will be at least 1 and not more than 100, and the number of bananas each trainer starts with will be a positive integer no more than 1073741823 (i.e. 2^30 -1). Some of them stockpile a LOT of bananas.



# --- Assignment Problem ---
# possible useful links:
# http://leeds-faculty.colorado.edu/glover/fred%20pubs/314%20-%20xQx%20-%20binary%20assignment%20with%20side%20constraints.pdf

# from scipy.optimize import linear_sum_assignment
import itertools

def solution(l):

    # The goal here is given a list, pair up each element in the list (essentially np.permutations) in a "cost" matrix, then by going through the upper triangle values, search for the MINIMUM amount of pairs with loops
    # If all pairs with loops can't be found, then start searching for no loops, until all elements are paired (even if some pairs have no loops)

    if (len(l) % 2 == 1):
        l.append(-1)

    # (1) Create an empty numpy matrix of size list * list, where list is the list you want to find the pairs / loops
    # You then want to fill the matrix with nans (I chose "nan" as a way to tell the algorithm not to access that part, but you could choose anything)

    matrix = np.empty((len(l), len(l)))
    matrix.fill(0)

    # (2) Go through the matrix and find the loops of the rows / columns, e.g. given the list [1, 7, 3, 21, 13, 19] and the matrix we just created where the rows / col are pairs of the list,
    # we place either 0 (representing a loop) and a 1 (representing a no loop), which generates a sort of "cost-pair" matrix. These numbers are arbitrary. Below is what the cost matrix would look like without findLoop():
    #       1       7       3       21      13      19
    # 1 [1,1]   [1,7]   [1,3]   [1,21]  [1,13]  [1,19]
    # 7 [7,1]   [7,7]   [7,3]   [7,21]  [7,13]  [7,19]
    # 3 [3,1]   [3,7]   [3,3]    ...     ...     ...
    # 21 ...     ...     ...     ...     ...     ... 
    # 13 ...     ...     ...     ...     ...     ... 
    # 19 ...     ...     ...     ...     ...     ...    
    # Given the example, lets say we look at index [0,3] which equates to pair [1,21], we calculate if this pair is a loop or not using the findLoop function (this is fairly inexpesnive). The cost matrix then looks like this:
    #       1   7   3  21  13  19
    # 1    [1   1   1   0   0   0]
    # 7    [1   1   0   1   0   0]
    # 3    [1   0   1   1   1   0]
    # 21   [0   1   1   1   0   0]
    # 13   [0   0   1   0   1   1]
    # 19   [0   0   0   0   1   1]
    # Notes:
    # Diagonal pairs are [x,x], which immediately equate to a no loop
    # With the max array size of 100, the maximum size cost matrix we would have is 100*100 -> 10,000, which is relatively small

    for i in range(0, len(l)):
        for j in range(0, len(l)):
                matrix[i][j] = findLoop(l[i], l[j])
    print(matrix)
    print(len(l))

    # (3) Finally do the linear_sum_assignment, which simply assigns the minimum pair in the cost matrix - we then sum this to find the minimum value overall
    # If you would like to know more, google the assignment problem or check the useful links above

    row_ind, col_ind = linear_sum_assignment(matrix)
    li = matrix[row_ind, col_ind]
    print(np.column_stack([row_ind, col_ind]))
    print(li)
    return int(li.sum())

# Decent way of finding out if a pair of 2 numbers leads to a loop or no loop, without any recursion / physically going through the pair step by step
# loop = 0
# no loop = 1
def findLoop(n, m):
    if (n == m or n == -1 or m == -1): return 1     # This is not allowed  
    x = bin(n + m)[2:]
    y = x.rstrip("0")
    k = int(len(x) - len(y))
    N = int(y, 2)
    if (k >= 2 and n % N == 0): return 1  # no loop = 1
    return 0    # loop = 0

# myList = [1, 7, 3, 21, 13, 19, 20]   # ODD NUMBER LIST FAILS for linear_sum_assignment
# myList = [1, 7, 3, 21, 13, 19]      # all loops, works
# myList = [1, 5]
# myList = [x for x in range(1, 100)]
# myList = np.random.randint(1, 2**30-1, 99)
# myList.sort()

# print(myList)
# start_time = timeit.default_timer()
# print(solution(myList))
# print("total time: " + str(timeit.default_timer() - start_time) + "\n")


# ---------------------- OLD RECURSION CODE----------------------

# # think about special case where the list passed is odd length, meaning we eventually will end up with a [[nan]] scenario
# # think about finding minimum value, my algorithm right now is a greedy backtracker
# def searchValidPairsInMatrix(m, l):

#     print("\noriginal List == " + str(l))

#     # exit clause (solution found)
#     if (m.size == 0):
#         print(sol)
#         return True
#         # return False  # enable this for all solutions

#     for findZero in np.argwhere(m == 0):
#         # print("choose a zero from the 'cost' matrix: " + str(findZero))
#         idx = list(set(range(m.shape[0])).difference(findZero))
#         reducedMatrix = m[np.ix_(idx,idx)]                                  # copy pasted code - essentially takes the difference of (cost matrix - original list) and finds what rows / cols to KEEP
#         reducedList = np.delete(l, [findZero[0], findZero[1]]).tolist()     # just as we reduce the cost matrix, reduce the actual list so we can find the pairs later in the algorithm
#         sol.append([l[x] for x in findZero])
#         print(str(sol) + " -> size == " + str(len(sol)))
#         print("remove the chosen zero row / col")
#         print(reducedMatrix)
#         if searchValidPairsInMatrix(reducedMatrix, reducedList):
#             return True
#         print("BACKTRACKING")
#         sol.pop()

#     # if no matching zero pairs are found, have to search with 1s now
#     # but this might not be efficient

#     return False

# ---------------------- OLD CODE----------------------

# percentage of finding a zero (in upper right triangle of matrix)
# allZeros = len(np.argwhere(m == 0))
# x = (len(m) * (len(m) + 1)) / 2
# y = (len(m) * (len(m) - 1)) / 2
# print(((allZeros - x) / y) * 100)

# def all_pairs(lst):
#     N = len(lst)
#     choice_indices = itertools.product(*[
#         xrange(k) for k in reversed(xrange(1, N, 2)) ])

#     for choice in choice_indices:
#         tmp = lst[:]
#         result = []
#         for index in choice:
#             result.append( (tmp.pop(0), tmp.pop(index)) )
#         yield result

# myList = []
# for x in all_pairs(testList):
#     print(x)
#     myList.append(x)
# print(len(myList))
# print(myList)

    # LOOP:
    # list1 = [1,4]
    # list1 = [4,10]
    # list1 = [5, 105]
    # list1 = [7,19]

    # NO LOOP:
    # list1 = [3,5]
    # list1 = [5,5]
    # list1 = [13,19]

# while(1):
#     print("findLoop:")
#     x = input("x = ")
#     y = input("y = ")
#     print(findLoop(x, y))

# print(solution(list1))

# for y in range(1, 2):
#     start = 1
#     # end = (2**y) - 1    # powers of 2: from 1 - 2^k + random number
#     end = 64
#     print("\ny == " + str(y)),
#     print("start == " + str(start)),
#     print(", end == " + str(end))
#     for start in range(start, end):
#         # if (solution([start, end-start]) == "NO LOOP"):
#         print(str([start, end-start]) + " == " + str(solution([start, end-start])))

# only for a list length of 2
# def recursion(l, t, h):
#     # (l[0] == l[1]) is special case where 2 numbers we started with are the same, therefore return immediately
#     if (l[0] == l[1]): return "same NO LOOP"

#     t = t[::-1] if t[0] > t[1] else t
#     t[1] -= t[0]
#     t[0] *= 2

#     print("t = " + str(t))
#     print("h = " + str(h))

#     # (t[0] == t[1]) checks if eventually the 2 numbers become the same, therefore NO LOOP
#     # (t == l) checks if eventually the temp array is equal to the original array, therefore LOOP (since order doesn't actually matter, we could check inverse of the array as well, but this is overkill, t == l is enough)
#     # (t in h) checks if temp is in history, therefore loop
#     if (t[0] == t[1]):
#         return "NO LOOP"
#     if (t == l):
#         return "LOOP"
#     if (t in h):
#         return "LOOP (HISTORY)"
#     h.append(t[:])
#     return recursion(l, t, h)

# l = [1, 1]
# print(recursion(l, l, []))



# ------------------------------------------------------------------------------------------------------------
            # LEVEL 4 (Q2)
# ------------------------------------------------------------------------------------------------------------



# Uh-oh -- you've been cornered by one of Commander Lambdas elite bunny trainers! Fortunately, you grabbed a beam weapon from an abandoned storeroom while you were running through the station, so you have a chance to fight your way out. But the beam weapon is potentially dangerous to you as well as to the bunny trainers: its beams reflect off walls, meaning you'll have to be very careful where you shoot to avoid bouncing a shot toward yourself!

# Luckily, the beams can only travel a certain maximum distance before becoming too weak to cause damage. You also know that if a beam hits a corner, it will bounce back in exactly the same direction. And of course, if the beam hits either you or the bunny trainer, it will stop immediately (albeit painfully).

# Write a function solution(dimensions, your_position, trainer_position, distance) that gives an array of 2 integers of the width and height of the room, an array of 2 integers of your x and y coordinates in the room, an array of 2 integers of the trainer's x and y coordinates in the room, and returns an integer of the number of distinct directions that you can fire to hit the elite trainer, given the maximum distance that the beam can travel.

# The room has integer dimensions [1 < x_dim <= 1250, 1 < y_dim <= 1250]. You and the elite trainer are both positioned on the integer lattice at different distinct positions (x, y) inside the room such that [0 < x < x_dim, 0 < y < y_dim]. Finally, the maximum distance that the beam can travel before becoming harmless will be given as an integer 1 < distance <= 10000.

# For example, if you and the elite trainer were positioned in a room with dimensions [3, 2], your_position [1, 1], trainer_position [2, 1], and a maximum shot distance of 4, you could shoot in seven different directions to hit the elite trainer (given as vector bearings from your location): [1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], and [-3, -2]. As specific examples, the shot at bearing [1, 0] is the straight line horizontal shot of distance 1, the shot at bearing [-3, -2] bounces off the left wall and then the bottom wall before hitting the elite trainer with a total shot distance of sqrt(13), and the shot at bearing [1, 2] bounces off just the top wall before hitting the elite trainer with a total shot distance of sqrt(5).



import numpy as np
import math

def solution(dim,yourpos,trainerpos,dist):

    # All the vectors from A -> B (fine apart from duplicates)
    xarray1 = np.sort(np.concatenate((np.arange(-yourpos[0] - trainerpos[0],1+dist,+2*dim[0]),np.arange(-yourpos[0] - trainerpos[0],-1-dist,-2*dim[0]))))
    xarray2 = np.sort(np.concatenate((np.arange(trainerpos[0] - yourpos [0],1+dist,+2*dim[0]),np.arange(trainerpos[0] - yourpos[0],-1-dist,-2*dim[0]))))
    yarray1 = np.sort(np.concatenate((np.arange(-yourpos[1] - trainerpos[1],1+dist,+2*dim[1]),np.arange(-yourpos[1] - trainerpos[1],-1-dist,-2*dim[1]))))
    yarray2 = np.sort(np.concatenate((np.arange(trainerpos[1] - yourpos[1],1+dist,+2*dim[1]),np.arange(trainerpos[1] - yourpos[1],-1-dist,-2*dim[1]))))
    a = np.array(np.meshgrid(np.unique(np.concatenate((xarray1, xarray2), axis=None)), np.unique(np.concatenate((yarray1, yarray2), axis=None)))).T.reshape(-1,2)   # permutations of x and y
    b = np.argwhere(np.hypot(a[:,0], a[:,1]) <= dist)  # all the indexes of "a" that are less than the distance given
    d = a[b][:,0]       # d is the array a which satisifies the distance condition - [:,0] else we have 3 brackets which is redundant

    # All the vectors from A -> A (same thing as above)
    xarray3 = np.sort(np.concatenate((np.arange(-yourpos[0] - yourpos[0],1+dist,+2*dim[0]),np.arange(-yourpos[0] - yourpos[0],-1-dist,-2*dim[0]))))
    xarray4 = np.sort(np.concatenate((np.arange(yourpos[0] - yourpos [0],1+dist,+2*dim[0]),np.arange(yourpos[0] - yourpos[0],-1-dist,-2*dim[0]))))
    yarray3 = np.sort(np.concatenate((np.arange(-yourpos[1] - yourpos[1],1+dist,+2*dim[1]),np.arange(-yourpos[1] - yourpos[1],-1-dist,-2*dim[1]))))
    yarray4 = np.sort(np.concatenate((np.arange(yourpos[1] - yourpos[1],1+dist,+2*dim[1]),np.arange(yourpos[1] - yourpos[1],-1-dist,-2*dim[1]))))
    a2 = np.array(np.meshgrid(np.unique(np.concatenate((xarray3, xarray4), axis=None)), np.unique(np.concatenate((yarray3, yarray4), axis=None)))).T.reshape(-1,2)
    b2 = np.argwhere(np.hypot(a2[:,0], a2[:,1]) <= dist)
    d2 = a2[b2][:,0]
    d2 = np.delete(d2, np.where(~d2.any(axis=1))[0], 0) # delete [0, 0] from matrices which can cause problems later for arctan2

    # print("\nA -> B")
    # print(d)
    # print("\nA -> A")
    # print(d2)

    # print("\n------------------------------")

    AtoB = np.array((np.sqrt(d[:,0]**2 + d[:,1]**2), np.arctan2(d[:,1], d[:,0]))).T
    # print(AtoB)
    AtoB = AtoB[np.lexsort((AtoB[:,0], AtoB[:,1]))]
    AtoBneg = np.array([i for i in AtoB if i[1] < 0])           # - ve
    AtoBpos = np.array([i for i in AtoB if i[1] >= 0])          # + ve
    # print("\nA -> B [distance, angle]")
    # print(AtoBneg)
    # print(AtoBpos)

    AtoA = np.array((np.sqrt(d2[:,0]**2 + d2[:,1]**2), np.arctan2(d2[:,1], d2[:,0]))).T
    AtoA = AtoA[np.lexsort((AtoA[:,0], AtoA[:,1]))]
    AtoAneg = np.array([i for i in AtoA if i[1] < 0])           # + ve
    AtoApos = np.array([i for i in AtoA if i[1] >= 0])          # - ve
    # print("\nA -> A [distance, angle]")
    # print(AtoAneg)
    # print(AtoApos)

    # print("\n------------------------------")

    # for AtoB, then join back to one array
    if AtoBneg.size != 0 and AtoBpos.size != 0:
        _, idx1_0 = np.unique(AtoBneg[:,1], return_index=True)              # - ve find duplicates along the 1st column (angle), [::-1] reverses order
        _, idx1_1 = np.unique(AtoBpos[:,1], return_index=True)              # + ve find duplicates along the 1st column (angle)
        AtoB = np.concatenate([AtoBneg[idx1_0], AtoBpos[idx1_1]])           # techincally we don't need to concatenate negative matrix in reverse order [::-1] but it looks nice
        # print("\nA -> B (removing duplicates and keeping minimum)")
        # print(AtoBneg[idx1_0])
        # print(AtoBpos[idx1_1])
        # print("\nconcatenate:")
        # print(AtoB)

    # for AtoA, then join back to one array
    if AtoAneg.size != 0 and AtoApos.size != 0:
        _, idx2_0 = np.unique(AtoAneg[:,1], return_index=True)              # - ve find duplicates along the 1st column (angle), [::-1] reverses order
        _, idx2_1 = np.unique(AtoApos[:,1], return_index=True)              # + ve find duplicates along the 1st column (angle)
        AtoA = np.concatenate([AtoAneg[idx2_0], AtoApos[idx2_1]])           # techincally we don't need to concatenate negative matrix in reverse order [::-1] but it looks nice
        # print("\nA -> A (removing duplicates and keeping minimum)")
        # print(AtoAneg[idx2_0])
        # print(AtoApos[idx2_1])
        # print("\nconcatenate:")
        # print(AtoA)

    # print("\n------------------------------")

    # comparing and finding the UNIQUE angle overlaps between A -> B and A -> A
    intersect = np.intersect1d(AtoB[:,1], AtoA[:,1])
    # print("\nintersection (same angle) of A -> B and A -> A")
    # print(intersect)
    if intersect.size != 0:
        AtoBcompare = AtoB[np.where(np.in1d(AtoB[:,1], intersect))[0]]
        AtoAcompare = AtoA[np.where(np.in1d(AtoA[:,1], intersect))[0]]
        # print("\nfind corresponding index of intersections")
        # print("A -> B")
        # print(AtoBcompare)
        # print("A -> A")
        # print(AtoAcompare)

        # if A -> B is less than A -> A, then we count this as a valid shooting angle, otherwise ignore
        # OR
        # if A -> B is greater than A -> A, delete from array
        delete = AtoBcompare[np.where(AtoBcompare[:,0] > AtoAcompare[:,0])]
        # print(delete)
        # print(np.where(AtoBcompare[:,0] > AtoAcompare[:,0]))
        for x in delete:
            # print("delete: " + str(x))
            AtoB = np.delete(AtoB, np.where((AtoB == x).all(axis=1)), axis=0)
    finalAnswer = AtoB

    # print(finalAnswer)
    return finalAnswer.shape[0]

# print(solution([3,2], [1,1], [2,1], 4))                 # test case 1 - should be 7
# print(solution([300,275], [150,150], [185,100], 500))   # test case 2 - should be 9
# print(solution([3,2], [1,1], [2,1], 20))                # random test case

# errors so far:
# 'AtoBcompare' is not the same length as 'AtoAcompare' when they should be, meaning we aren't getting unique values properly
# beam length is too small

# old code:
# AtoA = AtoA[AtoA[:,1].argsort()]      # old sorting method

# question doesn't specify what happens when you hit the corner when the angle isn't 45 degrees (see miro link for more details)



# ------------------------------------------------------------------------------------------------------------
            # LEVEL 5 (Q1)
# ------------------------------------------------------------------------------------------------------------



# Disorderly Escape
# =================

# Oh no! You've managed to free the bunny workers and escape Commander Lambdas exploding space station, but Lambda's team of elite starfighters has flanked your ship. If you dont jump to hyperspace, and fast, youll be shot out of the sky!

# Problem is, to avoid detection by galactic law enforcement, Commander Lambda planted the space station in the middle of a quasar quantum flux field. In order to make the jump to hyperspace, you need to know the configuration of celestial bodies in the quadrant you plan to jump through. In order to do *that*, you need to figure out how many configurations each quadrant could possibly have, so that you can pick the optimal quadrant through which youll make your jump. 

# There's something important to note about quasar quantum flux fields' configurations: when drawn on a star grid, configurations are considered equivalent by grouping rather than by order. That is, for a given set of configurations, if you exchange the position of any two columns or any two rows some number of times, youll find that all of those configurations are equivalent in that way -- in grouping, rather than order.

# Write a function solution(w, h, s) that takes 3 integers and returns the number of unique, non-equivalent configurations that can be found on a star grid w blocks wide and h blocks tall where each celestial body has s possible states. Equivalency is defined as above: any two star grids with each celestial body in the same state where the actual order of the rows and columns do not matter (and can thus be freely swapped around). Star grid standardization means that the width and height of the grid will always be between 1 and 12, inclusive. And while there are a variety of celestial bodies in each grid, the number of states of those bodies is between 2 and 20, inclusive. The solution can be over 20 digits long, so return it as a decimal string.  The intermediate values can also be large, so you will likely need to use at least 64-bit integers.

# For example, consider w=2, h=2, s=2. We have a 2x2 grid where each celestial body is either in state 0 (for instance, silent) or state 1 (for instance, noisy).  We can examine which grids are equivalent by swapping rows and columns.

# 00
# 00

# In the above configuration, all celestial bodies are "silent" - that is, they have a state of 0 - so any swap of row or column would keep it in the same state.

# 00 00 01 10
# 01 10 00 00

# 1 celestial body is emitting noise - that is, has a state of 1 - so swapping rows and columns can put it in any of the 4 positions.  All four of the above configurations are equivalent.

# 00 11
# 11 00

# 2 celestial bodies are emitting noise side-by-side.  Swapping columns leaves them unchanged, and swapping rows simply moves them between the top and bottom.  In both, the *groupings* are the same: one row with two bodies in state 0, one row with two bodies in state 1, and two columns with one of each state.

# 01 10
# 01 10

# 2 noisy celestial bodies adjacent vertically. This is symmetric to the side-by-side case, but it is different because there's no way to transpose the grid.

# 01 10
# 10 01

# 2 noisy celestial bodies diagonally.  Both have 2 rows and 2 columns that have one of each state, so they are equivalent to each other.

# 01 10 11 11
# 11 11 01 10

# 3 noisy celestial bodies, similar to the case where only one of four is noisy.

# 11
# 11

# 4 noisy celestial bodies.

# There are 7 distinct, non-equivalent grids in total, so solution(2, 2, 2) would return 7.

# -- Python cases --
# Input:
# solution.solution(2, 3, 4)
# Output:
#     430

# Input:
# solution.solution(2, 2, 2)
# Output:
#     7



import math
import fractions
import itertools

def solution(w, h, s):

    total = 0 

    for col_coeff, col_cycle in concatenate(partition(w)):
        for row_coeff, row_cycle in concatenate(partition(h)):

            combined = []

            for len_a, freq_a in col_cycle:
                for len_b, freq_b in row_cycle:

                    combined.append((len_a * len_b // fractions.gcd(len_a, len_b) , freq_a * freq_b * fractions.gcd(len_a , len_b)))

            value = 1
            combined = [sum(p[1] for p in combined)]

            for power in combined:
                value *= s ** power
                total += col_coeff * row_coeff * value

    return str(int(total)) 

# copy and paste from 'https://jeromekelleher.net/generating-integer-partitions.html' (there is a partition python module, but not in python 2.7)
# given a number, return all the different ways you can partition (split) that number, e.g. 3 -> [1+1+1, 1+2, 3]
def partitionNumber(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2 * x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

# inside the for loop below, the '[list(y) ... ] 'expression sort ints inside 'p' (the partition of the number) into their own lists so we can remove duplicates using set() - print the variables if still confused
# inside the 'yield' part, using a set to get unique values (i.e. remove duplicates), convert it to a list, then use [0] to return the int (we can do this because there is only ever going to be 1 element in the list due to set removing duplicates)
def formatPartition(p):
    for x in [list(y) for x, y in itertools.groupby(p)]:
        yield (list(set(x))[0], len(x))

# by using 'partitionNumber' to partition a given number, we then format each 'partition' element into: ( <length> , <number> ) - this is what 'formatPartition()' does
# e.g. 3 -> [1+1+1, 1+2, 3] then -> [3, 1] (three 1s), [1, 1] [1, 2] (one 1 and one 2), [1, 3] (one 3)
def partition(n):
    for par in partitionNumber(n):
        yield list(formatPartition(par))

# leo fraction code
def frac(list_tuple):
    for tup in list_tuple:
        val = 1 * math.factorial(tup[1]) * ( tup[0] ** tup[1])
    return Fraction(1, val) 

# final concatenation [ <fraction>, <partition> ]
def concatenate(partitionList):
    return [(frac(p), p) for p in partitionList]


print(concatenate(partition(4)))




# print(solution(5, 5, 3))

    






# helpful sources:
# https://stackoverflow.com/questions/48360864/nested-generators-and-yield-from     -->     regarding nested yields
# https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby          -->     itertools.groupby (using this to group partitions)

# ---------------------------------------------------------------

# old code

# (this is inside equivalentMatrices)
    # convertedPermuSize = flattenToReshape(permu, m.shape[0], m.shape[1] if len(m.shape) > 1 else 1)
    # print("\nprinting permutations:")
    # for i in convertedPermuSize:
    #     print(str(i) + "\n")

# def flattenToReshape(l, h, w):
#     for p in l:
#         yield np.array(p).reshape(h, w)

# matrix = np.zeros((1, 3))
# matrix[0, 0] = 1
# print(equivalentMatrices(matrix))

# ---------------------------------------------------------------

# naive method but wasting too much time so gave up
# in the 2x2 example, this works except it doesn't sort the diagonal / row / col cases, accounts for all, see miro for more details

# def equivalentMatrices(m):
#     print("matrix permutation of:\n" + str(m))

#     # this line below comes from: https://stackoverflow.com/questions/21959530/generate-all-unique-permutations-of-2d-array
#     # pprint(set(tuple([ ( (p[0],p[1]),(p[2],p[3]) ) for p in itertools.permutations([1,0,0,0]) ]) ))

#     # without "map tuple" code, cannot use "set" method to get unique values (see here: https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function/13464168)
#     permu = sorted(set(map(tuple, permutations(m))))

#     print(permu)

#     # print("\npermutation length:")
#     # print(sum(1 for x in convertedPermuSize))

#     return permu

# # yield offers superior memory efficiency when dealing with large permutations
# # m is matrix
# # duplicates array is essentially set() method but tailored for yield method
# def permutations(m):
#     for p in itertools.permutations(m):
#         yield [ele for ele in p]

# def createProductList(w, h, s):
#     for i in itertools.product(range(s), repeat = (w * h)):
#         yield i

# # find permutations of (n * m) with (s) states
# # e.g. (2,2) matrix with 2 states would be [0,0,0,0] , [0,0,0,1] ... [1,1,1,0] , [1,1,1,1]
# # length would be s^(n*m)

# width = 2
# height = 2
# states = 2

# productList = map(tuple, createProductList(width, height, states))
# pprint(productList)

# counter = 0
# removed = False
# while productList:
#     for ele in equivalentMatrices(productList[0]):
#         productList.remove(ele)
#         removed = True
#     if removed:
#         counter += 1
#         print(counter)
#         removed = False
#     pprint(productList, width=20)
# print(counter)

# ---------------------------------------------------------------

# complaints so far:

# - excuse me but why are you running python 2.7 (fixed with shebang anyway)
# - also why dont you tell us what the errors are in some of the test runs, e.g. we spent hours on one of the problems before and realised it was just a simple syntax error - spent like 3 hours finding it
# - LVL4 Q2 read bottom part (stuff about hitting corners at a non 45 degree angle)
# - SYMPY DOES NOT WORK ????????????????