#!/usr/bin/env python2.7

# Uh-oh -- you've been cornered by one of Commander Lambdas elite bunny trainers! Fortunately, you grabbed a beam weapon from an abandoned storeroom while you were running through the station, so you have a chance to fight your way out. But the beam weapon is potentially dangerous to you as well as to the bunny trainers: its beams reflect off walls, meaning you'll have to be very careful where you shoot to avoid bouncing a shot toward yourself!

# Luckily, the beams can only travel a certain maximum distance before becoming too weak to cause damage. You also know that if a beam hits a corner, it will bounce back in exactly the same direction. And of course, if the beam hits either you or the bunny trainer, it will stop immediately (albeit painfully).

# Write a function solution(dimensions, your_position, trainer_position, distance) that gives an array of 2 integers of the width and height of the room, an array of 2 integers of your x and y coordinates in the room, an array of 2 integers of the trainer's x and y coordinates in the room, and returns an integer of the number of distinct directions that you can fire to hit the elite trainer, given the maximum distance that the beam can travel.

# The room has integer dimensions [1 < x_dim <= 1250, 1 < y_dim <= 1250]. You and the elite trainer are both positioned on the integer lattice at different distinct positions (x, y) inside the room such that [0 < x < x_dim, 0 < y < y_dim]. Finally, the maximum distance that the beam can travel before becoming harmless will be given as an integer 1 < distance <= 10000.

# For example, if you and the elite trainer were positioned in a room with dimensions [3, 2], your_position [1, 1], trainer_position [2, 1], and a maximum shot distance of 4, you could shoot in seven different directions to hit the elite trainer (given as vector bearings from your location): [1, 0], [1, 2], [1, -2], [3, 2], [3, -2], [-3, 2], and [-3, -2]. As specific examples, the shot at bearing [1, 0] is the straight line horizontal shot of distance 1, the shot at bearing [-3, -2] bounces off the left wall and then the bottom wall before hitting the elite trainer with a total shot distance of sqrt(13), and the shot at bearing [1, 2] bounces off just the top wall before hitting the elite trainer with a total shot distance of sqrt(5).

# -----------------------------------------------------------------------------------------

# very badly coded, ran out of time

import numpy as np

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

print(solution([3,2], [1,1], [2,1], 4))                 # test case 1 - should be 7
print(solution([300,275], [150,150], [185,100], 500))   # test case 2 - should be 9
# print(solution([3,2], [1,1], [2,1], 20))                # random test case

# errors so far:
# 'AtoBcompare' is not the same length as 'AtoAcompare' when they should be, meaning we aren't getting unique values properly
# beam length is too small

# -----------------------------------------------------------------------------------------

# old code:
# AtoA = AtoA[AtoA[:,1].argsort()]      # old sorting method

# question doesn't specify what happens when you hit the corner when the angle isn't 45 degrees (see miro link for more details)