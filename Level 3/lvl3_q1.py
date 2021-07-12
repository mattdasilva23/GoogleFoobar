
# You're almost ready to make your move to destroy the LAMBCHOP doomsday device, but the security checkpoints that guard the underlying systems of the LAMBCHOP are going to be a problem. You were able to take one down without tripping any alarms, which is great! Except that as Commander Lambda's assistant, you've learned that the checkpoints are about to come under automated review, which means that your sabotage will be discovered and your cover blown -- unless you can trick the automated review system.

# To trick the system, you'll need to write a program to return the same security checksum that the bunny trainers would have after they would have checked all the workers through. Fortunately, Commander Lambda's desire for efficiency won't allow for hours-long lines, so the trainers at the checkpoint have found ways to quicken the pass-through rate. Instead of checking each and every worker coming through, the bunny trainers instead go over everyone in line while noting their worker IDs, then allow the line to fill back up. Once they've done that they go over the line again, this time leaving off the last worker. They continue doing this, leaving off one more worker from the line each time but recording the worker IDs of those they do check, until they skip the entire line, at which point they XOR the IDs of all the workers they noted into a checksum and then take off for lunch. Fortunately, the workers' orderly nature causes them to always line up in numerical order without any gaps.

# For example, if the first worker in line has ID 0 and the security checkpoint line holds three workers, the process would look like this:
# 0 1 2 /
# 3 4 / 5
# 6 / 7 8
# where the trainers' XOR (^) checksum is 0^1^2^3^4^6 == 2.

# Likewise, if the first worker has ID 17 and the checkpoint holds four workers, the process would look like:
# 17 18 19 20 /
# 21 22 23 / 24
# 25 26 / 27 28
# 29 / 30 31 32
# which produces the checksum 17^18^19^20^21^22^23^25^26^29 == 14.

# All worker IDs (including the first worker) are between 0 and 2000000000 inclusive, and the checkpoint line will always be at least 1 worker long.

# With this information, write a function solution(start, length) that will cover for the missing security checkpoint by outputting the same checksum the trainers would normally submit before lunch. You have just enough time to find out the ID of the first worker to be checked (start) and the length of the line (length) before the automatic review occurs, so your program must generate the proper checksum with just those two values.

# -----------------------------------------------------------------------------------------

import timeit

# xor of 1-n has pattern:
# if n % 4 == 0, then the xor or 1-n is n
# if n % 4 == 1, then the xor or 1-n is 1
# if n % 4 == 2, then the xor or 1-n is n + 1
# if n % 4 == 3, then the xor or 1-n is 0

# in this example:
# 17 18 19 20 /
# 21 22 23 / 24
# 25 26 / 27 28
# 29 / 30 31 32
# we want the left side triangle (17,18 etc.)

# using the rules above, to find the xor from 1-20 for example (from the 1st row), we do 20 % 4, which is 0, therefore xor from 1-20 == 0
# but we want 17-20 xor, so how do we do this? Well the interesting property of xor is if you do xor of 1-16 ^ 1-20 you actually get 17-20
# so for the first row, we get the starting number, minus 1, then find the end number in the row, and then use the rules above:
# 16 % 4 == 0, meaning xor from 1-16 = 16
# 20 % 4 == 0, meaning xor from 1-20 = 20 

def xor(n):
    # if n is 0 or negative, don't bother with calculation
    if n <= 0: return 0
    # xor rules for 1-n
    if (n % 4 == 0):
        x = n
    elif (n % 4 == 1):
        x = 1
    elif (n % 4 == 2):
        x = n + 1
    else:
        x = 0
    return x

def solution(s, l):
    answer = 0
    for x in range(l):
        firstNumInRow = ((s + (x*l)) - 1)       # find first number in specified row, -1 because if we are doing s-n, we need 1-(s-1), do note that this can be a -1 if s = 0
        lastNumInRow = (s + (l-1)) + (x*(l-1))  # find last number in specified row (x determines what row we are on)
        answer ^= (xor(firstNumInRow) ^ xor(lastNumInRow))
        # print(firstNumInRow, lastNumInRow)
        # print(answer)
    return answer

print("\n" + str(solution(17, 4)))
print("\n" + str(solution(0, 3)))

start_time = timeit.default_timer()
print("\n" + str(solution(2000000000, 10000)))
print("\ntotal time:")
print(timeit.default_timer() - start_time)

# -----------------------------------------------------------------------------------------

# old code

# def checkSum2(s, l):
#     arr = np.arange(s, s+(l*l))                 # create array from start number -> length of list
#     arr = np.reshape(arr, (l, l))               # reshape into 2d array
#     arr = np.flip(arr, 1)                       # flip the array to return correct partition of the triangle
#     arr = np.triu(arr, 0)                       # return triangle partition for xor
#     arr = arr.flatten()
#     return np.bitwise_xor.reduce(arr)           # reduce to calculate xor for all elements

# print(checkSum2(0, 3))
# print(checkSum2(17, 4))
# print(checkSum2(10, 10))

# start_time = timeit.default_timer()
# print(checkSum2(2000000000, 40000))
# print("total time:")
# print(timeit.default_timer() - start_time)
# print("\n")