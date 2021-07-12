#!/usr/bin/env python2.7

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

# -----------------------------------------------------------------------------------------


# from functools import reduce      # do we need this?
import math
import operator
import fractions
import itertools

# reference: https://jeromekelleher.net/generating-integer-partitions.html (there is a partition python module, but not in python 2.7)
# given a number, return all the different ways you can partition (split) that number, e.g. 3 -> [1+1+1,  1+2,3]
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
    return [(list(set(x))[0], len(x)) for x in [list(y) for x, y in itertools.groupby(p)]]

# by using 'partitionNumber' to partition a given number, we then format each 'partition' element into: ( <length> , <number> ) - this is what 'formatPartition()' does
# e.g. 3 -> [1+1+1, 1+2, 3] then -> [3, 1] (three 1s), [1, 1] [1, 2] (one 1 and one 2), [1, 3] (one 3)
def partition(n):
    return [list(formatPartition(par)) for par in partitionNumber(n)]

# product function used for 'frac'
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

# finding fraction coefficient
def frac(list_tuple):
    return fractions.Fraction(1, prod([math.factorial(tup[1]) * (tup[0] ** tup[1]) for tup in list_tuple]))  

# final concatenation [ <fraction>, <partition> ]
# e.g. [(Fraction(1, 24), [(1, 4)]), ... ]
def concatenate(partitionList):
    return [(frac(p), p) for p in partitionList]

# create cycle index
def cycle_index(n):
    return concatenate(partition(n))

# unreadable one liner
def createProdIndex(monom1, monom2):
    return [sum(p[1] for p in [(sub1 * sub2 // fractions.gcd(sub1, sub2) , sup1 * sup2 * fractions.gcd(sub1 , sub2)) for sub1, sup1 in monom1 for sub2, sup2 in monom2])][0]

# another unreadable one liner
def solution(w, h, s):
    return str(sum([coeff1 * coeff2 * (s ** createProdIndex(monom1, monom2)) for coeff1, monom1 in cycle_index(w) for coeff2, monom2 in cycle_index(h)]))

# print(concatenate(partition(4)))

print(solution(2, 2, 2))        # should return 7
# print(solution(2, 3, 4))        # should return 430
# print(solution(5, 5, 3))        # should return 64796982
# print(solution(20, 20, 20))     # should return 4362636532842096060384943807744636125276397960499827368129084188447932018978056642477757458957608927359934657503098196804450318141141777881011615585512957407760163943909711433512752447285147752964634227190809004820890439784029740040762393940714469208411020089723197537701566510883455745253715100731646069527339605771869192201854698546299608913490394400810614096581276817661724939338918115013630804102540735299259439242860527025876886776519058749670961452511463618715531904197385737000 (pretty big number)

# ---------------------------------------------------------------

# our sympy solution below, however python 2.7 does not support sympy, therefore we resorted to using code above

# from sympy import symbols
# from fractions import Fraction

# def indexpoly(n):
#     return sympy.simplify(sum([Fraction(1, n) * sympy.Function('s')(k) * indexpoly(n-k) for k in range(1, n+1)])) if n != 0 else 1

# helpful sources:
# https://stackoverflow.com/questions/48360864/nested-generators-and-yield-from     -->     nested yields
# https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby            -->     itertools.groupby (using this to group partitions)
# https://realpython.com/python-reduce-function/                                    -->     reduce function

# ---------------------------------------------------------------

# readable version of code

# import math
# import fractions
# import itertools

# def solution(w, h, s):
#     total = 0 
#     for col_coeff, col_cycle in concatenate(partition(w)):
#         for row_coeff, row_cycle in concatenate(partition(h)):
#             combined = []
#             for len_a, freq_a in col_cycle:
#                 for len_b, freq_b in row_cycle:
#                     combined.append((len_a * len_b // fractions.gcd(len_a, len_b) , freq_a * freq_b * fractions.gcd(len_a , len_b)))
#             value = 1
#             combined = [sum(p[1] for p in combined)]
#             for power in combined:
#                 value *= s ** power
#                 total += col_coeff * row_coeff * value
#     return str(int(total)) 

# # copy and paste from 'https://jeromekelleher.net/generating-integer-partitions.html' (there is a partition python module, but not in python 2.7)
# # given a number, return all the different ways you can partition (split) that number, e.g. 3 -> [1+1+1, 1+2, 3]
# def partitionNumber(n):
#     a = [0 for i in range(n + 1)]
#     k = 1
#     y = n - 1
#     while k != 0:
#         x = a[k - 1] + 1
#         k -= 1
#         while 2 * x <= y:
#             a[k] = x
#             y -= x
#             k += 1
#         l = k + 1
#         while x <= y:
#             a[k] = x
#             a[l] = y
#             yield a[:k + 2]
#             x += 1
#             y -= 1
#         a[k] = x + y
#         y = x + y - 1
#         yield a[:k + 1]

# # inside the for loop below, the '[list(y) ... ] 'expression sort ints inside 'p' (the partition of the number) into their own lists so we can remove duplicates using set() - print the variables if still confused
# # inside the 'yield' part, using a set to get unique values (i.e. remove duplicates), convert it to a list, then use [0] to return the int (we can do this because there is only ever going to be 1 element in the list due to set removing duplicates)
# def formatPartition(p):
#     for x in [list(y) for x, y in itertools.groupby(p)]:
#         yield (list(set(x))[0], len(x))

# # by using 'partitionNumber' to partition a given number, we then format each 'partition' element into: ( <length> , <number> ) - this is what 'formatPartition()' does
# # e.g. 3 -> [1+1+1, 1+2, 3] then -> [3, 1] (three 1s), [1, 1] [1, 2] (one 1 and one 2), [1, 3] (one 3)
# def partition(n):
#     for par in partitionNumber(n):
#         yield list(formatPartition(par))

# def frac(list_tuple):
#     val = 1
#     for tup in list_tuple:
#         val = val * math.factorial(tup[1]) * ( tup[0] ** tup[1])
#     return fractions.Fraction(1, val) 

# # final concatenation [ <fraction>, <partition> ]
# def concatenate(partitionList):
#     return [(frac(p), p) for p in partitionList]

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

# naive method but wasting too much time so gave up lol
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