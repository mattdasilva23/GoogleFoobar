#!/usr/bin/env python2.7

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

# -----------------------------------------------------------------------------------------

# --- Assignment Problem ---
# possible useful links:
# http://leeds-faculty.colorado.edu/glover/fred%20pubs/314%20-%20xQx%20-%20binary%20assignment%20with%20side%20constraints.pdf

from scipy.optimize import linear_sum_assignment
import numpy as np
import timeit

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
myList = [1, 7, 3, 21, 13, 19]      # all loops, works
# myList = [1, 3]
# myList = [x for x in range(1, 100)]
# myList = list(np.random.randint(1, 2**30-1, 99))
myList.sort()

print(myList)
start_time = timeit.default_timer()
print(solution(myList))
print("total time: " + str(timeit.default_timer() - start_time) + "\n")

# -----------------------------------------------------------------------------------------

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