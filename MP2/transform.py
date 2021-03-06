
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    limit = arm.getArmLimit()
    dim_x = int((max(limit[0]) - min(limit[0])) / granularity + 1)
    dim_y = int((max(limit[1]) - min(limit[1])) / granularity + 1)
    map = [[" "] * dim_y for _ in range(dim_x)]
    startIdx = angleToIdx(arm.getArmAngle(), [min(limit[0]), min(limit[1])], granularity)
    # map[startIdx[0]][startIdx[1]] = 'P'
    for i in range(dim_x):
        for j in range(dim_y):
            alpha, beta = idxToAngle([i, j], [min(limit[0]), min(limit[1])], granularity)
            arm.setArmAngle((alpha, beta))
            if doesArmTipTouchGoals(arm.getEnd(), goals):
                map[i][j] = '.'
            elif doesArmTouchObjects(arm.getArmPosDist(), obstacles, False):
                map[i][j] = '%'
            elif doesArmTouchObjects(arm.getArmPosDist(), goals, True):
                map[i][j] = '%'
            elif not isArmWithinWindow(arm.getArmPos(), window):
                map[i][j] = '%'
    map[startIdx[0]][startIdx[1]] = 'P'
    # print(len(map), len(map[0]))
    myMaze = Maze(map, (min(limit[0]), min(limit[1])), granularity)
    return myMaze
