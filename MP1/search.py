# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from queue import PriorityQueue
from collections import defaultdict
import math
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def sanity_check(maze, path):
    """
    Runs check functions for part 0 of the assignment.

    @param maze: The maze to execute the search on.
    @param path: a list of tuples containing the coordinates of each state in the computed path

    @return bool: whether or not the path pass the sanity check
    """
    # TODO: Write your code here
    if maze.isValidPath(path) == "Valid":
        return True
    return False


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    queue = [maze.getStart()]
    prev = {queue[0]:None}
    end = None
    while queue:
        curr = queue.pop(0)
        if maze.isObjective(curr[0], curr[1]):
            end = curr
            break
        neighbors = maze.getNeighbors(curr[0], curr[1])
        for n in neighbors:
            if n not in prev:
                prev[n] = curr
                queue.append(n)
    path = []
    while end:
        path = [end] + path
        end = prev[end]
    return path

class SimpleAstar:
    def __init__(self, maze, start, obj):
        self.path = []
        boundary = PriorityQueue()
        prev = {}
        dist = {}
        prev[start] = None
        dist[start] = 0
        manhatten = abs(obj[0] - start[0]) + abs(obj[1] - start[1])
        boundary.put((manhatten + 0, start))
        end = None
        while not boundary.empty():
            value, curr = boundary.get()
            if curr == obj:
                end = curr
                break
            neighbors = maze.getNeighbors(curr[0], curr[1])
            for n in neighbors:
                if n in prev:
                    continue
                prev[n] = curr
                dist[n] = dist[curr] + 1
                manhatten = abs(obj[0] - n[0]) + abs(obj[1] - n[1])
                boundary.put((manhatten + dist[n], n))
        while end:
            self.path = [end] + self.path
            end = prev[end]

    def getPath(self):
        return self.path

    def getCost(self):
        return len(self.path) - 1

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    objs =  maze.getObjectives()
    MyAstar = SimpleAstar(maze, maze.getStart(), objs[0])
    return MyAstar.getPath()

class Topology:
    def __init__(self, maze, objs):
        self.savedCosts = {}
        self.savedPaths = {}
        self.MSTCache = {}
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                tmpAstar = SimpleAstar(maze, objs[i], objs[j])
                self.savedCosts[(objs[i], objs[j])] = tmpAstar.getCost()
                self.savedPaths[(objs[i], objs[j])] = tmpAstar.getPath()

    def getPath(self, start, obj):
        if start == obj:
            return [start]
        if (start, obj) in self.savedPaths:
            return self.savedPaths[(start, obj)]
        elif (obj, start) in self.savedPaths:
            return self.savedPaths[(obj, start)]
        else:
            return None

    def getCost(self, start, obj):
        if start == obj:
            return 0
        if (start, obj) in self.savedPaths:
            return self.savedCosts[(start, obj)]
        elif (obj, start) in self.savedPaths:
            return self.savedCosts[(obj, start)]
        else:
            return math.inf

    def MST(self, start, objs):
        if not objs:
            return 0
        objs = sorted(objs, key = lambda x:x[1])
        objs = sorted(objs, key = lambda x:x[0])
        if tuple(objs) in self.MSTCache:
            return self.MSTCache[tuple(objs)]
        selected = [objs[0]]
        candidate = objs[1:]
        cost = 0
        while candidate:
            newSeleted, minCost = None, math.inf
            for node1 in selected:
                for node2 in candidate:
                    if self.getCost(node1, node2) < minCost:
                        minCost = self.getCost(node1, node2)
                        newSeleted = node2
            selected.append(newSeleted)
            candidate.remove(newSeleted)
            cost += minCost
        self.MSTCache[tuple(objs)] = cost
        dist = math.inf
        for obj in objs:
            dist = min(dist, self.getCost(start, obj))
        return cost + dist


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    graph = Topology(maze, [maze.getStart()] + maze.getObjectives())
    start = maze.getStart()
    objs = maze.getObjectives()
    dist = {start:0}
    order = [start]
    while objs:
        boundary = PriorityQueue()
        for obj in objs:
            tmp = objs.copy()
            tmp.remove(obj)
            boundary.put((graph.MST(obj, tmp) + graph.getCost(order[-1], obj), obj))
            del tmp
        value, curr = boundary.get()
        objs.remove(curr)
        order.append(curr)
        del boundary
    path = [start]
    for i in range(len(order) - 1):
        tour = graph.getPath(order[i], order[i + 1])
        if tour[0] != order[i]:
            tour[:] = tour[::-1]
        path += tour[1:]
    return path

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    path = []
    boundary = PriorityQueue()
    start = maze.getStart()
    objs = maze.getObjectives()
    prev = {}
    dist = {}
    prev[start] = None
    dist[start] = 0
    manhatten = abs(obj[0] - start[0]) + abs(obj[1] - start[1])
    boundary.put((manhatten + 0, start))
    end = None
    while not boundary.empty():
        value, curr = boundary.get()
        if curr == obj:
            end = curr
            break
        neighbors = maze.getNeighbors(curr[0], curr[1])
        for n in neighbors:
            if n in prev:
                continue
            prev[n] = curr
            dist[n] = dist[curr] + 1
            manhatten = abs(obj[0] - n[0]) + abs(obj[1] - n[1])
            boundary.put((manhatten + dist[n], n))
    while end:
        self.path = [end] + self.path
        end = prev[end]
    return


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
