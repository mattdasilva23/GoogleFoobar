#!/usr/bin/env python2.7

# Commander Lambda has asked for your help to refine the automatic quantum antimatter fuel injection system for the LAMBCHOP doomsday device. It's a great chance for you to get a closer look at the LAMBCHOP -- and maybe sneak in a bit of sabotage while you're at it -- so you took the job gladly. 

# Quantum antimatter fuel comes in small pellets, which is convenient since the many moving parts of the LAMBCHOP each need to be fed fuel one pellet at a time. However, minions dump pellets in bulk into the fuel intake. You need to figure out the most efficient way to sort and shift the pellets down to a single pellet at a time. 

# The fuel control mechanisms have three operations: 

# 1) Add one fuel pellet
# 2) Remove one fuel pellet
# 3) Divide the entire group of fuel pellets by 2 (due to the destructive energy released when a quantum antimatter pellet is cut in half, the safety controls will only allow this to happen if there is an even number of pellets)

# Write a function called solution(n) which takes a positive integer as a string and returns the minimum number of operations needed to transform the number of pellets to 1. The fuel intake control panel can only display a number up to 309 digits long, so there won't ever be more pellets than you can express in that many digits.

# For example:
# solution(4) returns 2: 4 -> 2 -> 1
# solution(15) returns 5: 15 -> 16 -> 8 -> 4 -> 2 -> 1

# -----------------------------------------------------------------------------------------

import timeit
# import sys

# idea here is to convert to binary, then depending on if the digit is 1 or 0, divide by 2 or add/subtract 1
# how do we know when to add/subtract 1? when we have trailing 1s e.g. 511 in binary is 111111111, we want to add 1 so it carries over
# to all the other 1s in the number, so it because 1000000000 (512) and therefore we can just divide by 2 now to get to 1
# when the next character is NOT a 1, we just add 1 so it becomes even and continue the loop/recursion

def solution(s):
    s = long(s)
    steps = 0
    while(s != 1):
        binary = long(bin(s)[2:])       # represent s with binary (easier to calculate)
        if (binary % (10) == 0):
            s //= 2                     # do integer division here because we get a float index error otherwise
        else:
            if ((binary//10) % 10 == 1 and not s == 3):  # 3 is exception case
                s += 1
            else:
                s -= 1
        steps += 1
        # print(s, binary, steps)
    return steps

start_time = timeit.default_timer()
print(solution("203956878356401977405765866929034577280193993314348263094772646453283062722701277632936616063144088173312372882677123879538709400158306567338328279154499698366071906766440037074217117805690872792848149112022286332144876183376326512083574821647933992961249917319836219304274280243803104015000563790123191919191"))    # this is max 309 digit length
print("\ntotal time:")
print(timeit.default_timer() - start_time)

# -----------------------------------------------------------------------------------------

# # old code which uses recursion (this actually works fine but we just get a recursion depth error cos the digits are too big, so change that with the code below)
# sys.setrecursionlimit(2000)

# def recursion(s, steps):
#     binary = int(bin(s)[2:])    # represent s with binary (easier to calculate)
#     if (s == 1): return steps   # exit clause
#     steps += 1
#     print(s, binary)
#     if (binary % 10 == 0):
#         s /= 2
#     else:
#         s += 1 if ((binary/10) % 10 == 1 and not s == 3) else -1  # if next char is 1, add 1, else -1, (3 is exception case where we always want to -1)
#     return recursion(s, steps)

# def solution(s):
#     return recursion(int(s), 0)

# start_time = timeit.default_timer()
# print(solution("15"))    # worse case scenario (101010101010101 etc...)
# print("total time:")
# print(timeit.default_timer() - start_time)
# print("\n")

# start_time = timeit.default_timer()
# print(solution("234074626935197383818919946717320928856507419133112834991445418174131088288412712412380829846884812527500148281082543434451549178143771643870895091978481932783551944563626933953518515105789185051371201760527351261508008700966748543585824362683368798389779310787221742499931309815728797271921036887531541845"))    # worse case scenario (101010101010101 etc...)
# print("total time:")
# print(timeit.default_timer() - start_time)
# print("\n")

# start_time = timeit.default_timer()
# print(solution("203956878356401977405765866929034577280193993314348263094772646453283062722701277632936616063144088173312372882677123879538709400158306567338328279154499698366071906766440037074217117805690872792848149112022286332144876183376326512083574821647933992961249917319836219304274280243803104015000563790123191919191"))    # this is max 309 digit length
# print("total time:")
# print(timeit.default_timer() - start_time)