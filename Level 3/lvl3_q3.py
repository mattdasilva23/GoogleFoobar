#!/usr/bin/env python2.7

# Making fuel for the LAMBCHOP's reactor core is a tricky process because of the exotic matter involved. It starts as raw ore, then during processing, begins randomly changing between forms, eventually reaching a stable form. There may be multiple stable forms that a sample could ultimately reach, not all of which are useful as fuel. 

# Commander Lambda has tasked you to help the scientists increase fuel creation efficiency by predicting the end state of a given ore sample. You have carefully studied the different structures that the ore can take and which transitions it undergoes. It appears that, while random, the probability of each structure transforming is fixed. That is, each time the ore is in 1 state, it has the same probabilities of entering the next state (which might be the same state).  You have recorded the observed transitions in a matrix. The others in the lab have hypothesized more exotic forms that the ore can become, but you haven't seen all of them.

# Write a function solution(m) that takes an array of array of nonnegative ints representing how many times that state has gone to the next state and return an array of ints for each terminal state giving the exact probabilities of each terminal state, represented as the numerator for each state, then the denominator for all of them at the end and in simplest form. The matrix is at most 10 by 10. It is guaranteed that no matter which state the ore is in, there is a path from that state to a terminal state. That is, the processing will always eventually end in a stable state. The ore starts in state 0. The denominator will fit within a signed 32-bit integer during the calculation, as long as the fraction is simplified regularly. 

# For example, consider the matrix m:
# [
#   [0,1,0,0,0,1],  # s0, the initial state, goes to s1 and s5 with equal probability
#   [4,0,0,3,2,0],  # s1 can become s0, s3, or s4, but with different probabilities
#   [0,0,0,0,0,0],  # s2 is terminal, and unreachable (never observed in practice)
#   [0,0,0,0,0,0],  # s3 is terminal
#   [0,0,0,0,0,0],  # s4 is terminal
#   [0,0,0,0,0,0],  # s5 is terminal
# ]
# So, we can consider different paths to terminal states, such as:
# s0 -> s1 -> s3
# s0 -> s1 -> s0 -> s1 -> s0 -> s1 -> s4
# s0 -> s1 -> s0 -> s5
# Tracing the probabilities of each, we find that
# s2 has probability 0
# s3 has probability 3/14
# s4 has probability 1/7
# s5 has probability 9/14
# So, putting that together, and making a common denominator, gives an answer in the form of
# [s2.numerator, s3.numerator, s4.numerator, s5.numerator, denominator] which is
# [0, 3, 2, 9, 14].

# -----------------------------------------------------------------------------------------

# absorbing markov chain

# BIG NOTE: FOR FLOATING POINT DIVISION IN PYTHON 2.7, ADD THIS: from __future__ import division
# print( ((1/2)*(3/9)) / (1 - ((1/2)*(4/9))) ) # should equal 3/14 which is probability of S0 -> S3

from __future__ import division
import numpy as np
from fractions import Fraction

def probabilityConversion(matrix):
    len = matrix.shape[0]
    x = np.zeros((len,1))
    for i in range(len):
        x[i] = sum(matrix[i,:])
        if (x[i] != 0):
            x[i] = 1 / int(x[i])
    matrix = np.multiply(matrix,x)
    return matrix

def solution(m):

    m = np.array(m)
    if m.size == 1:
            return [1,1]
    nonTerminal = np.argwhere(np.sum(m, axis=1) != 0)
    terminal = np.argwhere(np.sum(m, axis=1) == 0)
    if (len(terminal) > 1):
        m = probabilityConversion(m)
    new_order = np.array(list(nonTerminal) + list(terminal)).flatten()
    fraction = np.array([[m[i][j] for j in new_order] for i in new_order])

    # print("fraction1: " + str(fraction))
    # print("fraction: " + str(fraction))

    # fraction is reordered probability matrix, nonTerminal is the list of non terminal states
    Q = fraction[:len(nonTerminal),:len(nonTerminal)]
    size = Q.shape[0]                           # find size of the matrix Q
    I = np.identity(size)                       # Define identity matrix by taking the size of Q
    N = np.linalg.inv(I-Q)                      # Defining Fundamental matrix N
    R = fraction[0:len(nonTerminal),len(nonTerminal):]

    # print("Size: " + str(size))
    # print("Q: " + str(Q))
    # print("I: " + str(I))
    # print("N: " + str(N))
    # print("R: " + str(R))
    B = np.matmul(N,R)[0,:]    # defining B = NR
    B = [Fraction(B[i]).limit_denominator() for i in range(len(B))]
    # print("B: " + str(B))

    
    LCM = np.lcm.reduce([f.denominator for f in B])   # find LCM
    
    vals = [int(f.numerator * LCM / f.denominator) for f in B]
    vals.append(LCM)

    # print(vals)
    # print(LCM)
    # print(B)

    return vals

print(solution([
    [0,1,0,0,0,1],  # s0, the initial state, goes to s1 and s5 with equal probability
    [4,0,0,3,2,0],  # s1 can become s0, s3, or s4, but with different probabilities
    [0,0,0,0,0,0],  # s2 is terminal, and unreachable (never observed in practice)
    [0,0,0,0,0,0],  # s3 is terminal
    [0,0,0,0,0,0],  # s4 is terminal
    [0,0,0,0,0,0],  # s5 is terminal
    ]))

# matrix = ( [[0,1,0,0,0,1],
#             [4,0,0,3,2,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0],
#             [0,0,0,0,0,0], ])

# quad loop example (with no terminator)
# matrix = np.array( [[0,1,0,0],
#                     [0,0,1,0],
#                     [0,0,0,1],
#                     [1,0,0,0], ])

# matrix = np.array( [[0,1,0,0],
#                     [2,0,3,4],
#                     [0,5,0,0],
#                     [0,0,0,0], ])

# matrix = np.array( [[0,0.5],
#                     [0,0],])

# matrix = np.array( [[0,8,3,2,0,0],
#                     [2,0,5,0,5,0],
#                     [0,3,0,6,2,0],
#                     [5,0,0,0,0,9],
#                     [0,0,1,0,0,0],
#                     [0,0,0,0,0,0], ])

# matrix = np.array( [[0,1,0,0,0],
#                     [0,0,1,1,0],
#                     [1,0,0,0,0],
#                     [0,1,0,0,1],
#                     [0,0,0,0,0], ])

# matrix = np.array( [[0,1,0,0,0],
#                     [0,0,1,1,0],
#                     [0,1,0,1,1],
#                     [1,0,0,1,1],
#                     [0,0,0,0,0], ])

# print("\n")
# print(solution(matrix))

# -----------------------------------------------------------------------------------------

# OLD BAD CODE USING DFS

# visited = []
# loop = []
# terminal = []

# def solution(m):
#     DFS(matrix, 0)
#     print("\nvisited rows (states): " + str(visited))
#     print("loops found (states): " + str(loop))
#     print("terminals found (states): " + str(terminal))
#     return -1

# def DFS(matrix, row):
#     print("\n\n")
#     if visited:
#         print("went from S" + str(visited[-1]) + " to"),
#     print("S" + str(row) + ": " + str(matrix[row]))

#     if (matrix[row].sum() == 0 and row not in terminal):
#         print("\tterminal state found at: S" + str(row))
#         terminal.append(row)
#         return

#     if (row in visited):
#         print("\tloop found:"),
#         temp = []
#         for x in visited[::-1]:
#             print(" S" + str(x) + " ->"),
#             temp.append(x)
#             print()
#             print("\n\t\tvisited rows (states): " + str(visited))
#             print("\t\tloops found (states): " + str(loop))
#             print("\t\ttemp: " + str(temp[::-1]))
#             print("\t\tx: " + str(x)),
#             print("== row: " + str(row))
#             if (temp[::-1] in loop and x == row):
#                 print("\t\tALREADY FOUND THIS LOOP")
#                 return
#             if (x == row):
#                 break
#         if (temp[::-1] in loop):
#             print("ALREADY FOUND THIS LOOP")
#             return
#         loop.append(temp[::-1])
#         print("")
#         del visited[:]
#         return

#     for x, v in enumerate(matrix[row]):
#         # print(x, v)
#         if (v == 0):
#             continue
#         else:
#             if row not in visited:
#                 visited.append(row)
#             print("\nvisited rows (states): " + str(visited))
#             print("loops found (states): " + str(loop))
#             print("terminals found (states): " + str(terminal))
#             DFS(matrix, x)