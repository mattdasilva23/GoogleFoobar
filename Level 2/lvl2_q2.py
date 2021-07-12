#!/usr/bin/env python2.7

from bisect import bisect

# old way which was:
    # rightindexes = [i for i,x in enumerate(s) if x == '>']
    # leftindexes =  [i for i,x in enumerate(s) if x == '<']
# is slightly slower because there are 2 for loops in the above case

# bisect method runs O(log(n)) (there are 2 bisect methods, left and right, either does not matter in this case)
# bisect method explanation:
# - takes 2 inputs (array, x) where array is a sorted array to search and x is the number to locate 
# - uses bisection (binary search) to find the correct position of where to place x in the sorted array
# - then simply returns that index, which just so happens to be the exact number of elements in the array that are less than the given number
# the reason bisect left / right does not matter in this case is because you only need to use it if the element is the same e.g.
# [0,1,2], and you are searching for 1, but 1 is already in the array so what index do you return? left will return the index
# 1 to the left of the number found (0), and right will return the index 1 to the right found (2)
# in our case, we will never have a duplicate number, we sort the elements in either right or left list

# the idea of this method is to sort them into right and left lists and then using the left array, find how many times the left salutes the right
# e.g. "<<>><" - the first char represents a left soldier going left, he does not pass any right going soldiers, therefore does not salute
# the 2 right soldiers pass 1 left soldier (end of array) therefore 2 salutes happen, but we didn't take into account the left saluting the right
# on the way back, therefore a total of 4 salutes happen. We can optimise this by not searching both ways, just one way and doubling the output
# so in the code below, we search the left side going right and see how many times they pass a right soldier

# worst case: any combination of left against right, e.g. <>, <<<>>>, <<<<<>>>>>, etc. since no salutes are found but the algorithm will still search
# best case: no real best case, just a bunch of dashes

def solution(s):
    s = s.replace("-","")
    rightindexes = []
    leftindexes = []
    [rightindexes.append(i) if x == ">" else leftindexes.append(i) for i, x in enumerate(s)]
    return 2*sum([bisect(rightindexes, i) for i in leftindexes])

print(solution("--->-><-><-->-"))   # should return 10
print(solution("<<>><"))            # should return 4

# print(solution("<><><><><><><><>"))
# print(solution(">>>>>>>><<<<<<<<"))
# print(solution("<<<<<<<<>>>>>>>>"))

# -----------------------------------------------------------------------------------------

# debug code:
# def solution(s):
#     s = s.replace("-","")
#     rightindexes = []
#     leftindexes = []
#     print("\nstring: " + str(s) + ", length: " + str(len(s)))
#     for i, x in enumerate(s):
#         print("position: " + str(i) + ", " + str(s[i]) + " - " + ("right" if x == ">" else "left"))
#         rightindexes.append(i) if x == ">" else leftindexes.append(i)
#     print("right\t" + str(rightindexes))
#     print("left\t" + str(leftindexes))
#     count = 0
#     print("\nfind indexes(a.k.a. less than):")
#     for i in leftindexes:
#         count += (bisect(rightindexes, i))
#         print("bisect(" + str(rightindexes) + ", " + str(i) + ") == " + str(bisect(rightindexes, i)))
#     return int(2*count)