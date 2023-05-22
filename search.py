# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # initialize the search tree using initial state of the problem
    fringe = util.Stack()
    ways = util.Stack()
    currState = problem.getStartState()
    fringe.push(currState)
    list1 = []
    ways.push(list1)
    expandedNodes = []
    while True:
        if fringe.isEmpty():
            return None
        currState = fringe.pop()
        if problem.isGoalState(currState):
            list1 = ways.pop()
            return list1
        successors = problem.getSuccessors(currState)
        expandedNodes.append(currState)
        tempList = ways.pop()
        for i in range(len(successors)):
            if not (successors[i][0] in expandedNodes):
                fringe.push(successors[i][0])
                copyList = list.copy(tempList)
                copyList.append(successors[i][1])
                ways.push(copyList)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # initialize the search tree using initial state of the problem
    fringe = util.Queue()
    ways = util.Queue()
    currState = problem.getStartState()
    fringe.push(currState)
    list1 = []
    ways.push(list1)
    fringeNodes = []
    fringeNodes.append(currState)
    while True:
        if fringe.isEmpty():
            return None
        currState = fringe.pop()
        if problem.isGoalState(currState):
            return ways.pop()
        successors = problem.getSuccessors(currState)
        tempList = ways.pop()
        for i in range(len(successors)):
            if not (successors[i][0] in fringeNodes):
                fringeNodes.append(successors[i][0])
                fringe.push(successors[i][0])
                copyList = list.copy(tempList)
                copyList.append(successors[i][1])
                ways.push(copyList)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # initialize the search tree using initial state of the problem
    fringe = util.PriorityQueue()
    currState = [[], [], []]
    currState[0].append(problem.getStartState())
    currState[2].append(0)

    fringe.push(item=currState, priority=currState[2][0])
    fringeNodes = []
    # uncomment the comment below and comments in lines 177-180 if you want to change ucs to dfs,
    # cost = 1
    while True:
        if fringe.isEmpty():
            return None
        currState = fringe.pop()
        if currState[0][0] in fringeNodes:
            continue
        fringeNodes.append(currState[0][0])
        if problem.isGoalState(currState[0][0]):
            return currState[1]
        successors = problem.getSuccessors(currState[0][0])
        for i in range(len(successors)):
            if not (successors[i][0] in fringeNodes):
                newNode = [[]]
                newNode.append([])
                newNode.append([])
                newNode[0].append(successors[i][0])
                newNode[1] = list(currState[1]).copy()
                newNode[1].append(successors[i][1])
                newNode[2].append(int(currState[2][0])
                                  + successors[i][2])

                # how to convert ucs to dfs, remember to also uncomment line 155
                # newNode[2].append(int(currState[2][0])
                #                   + cost)
                # cost /= 2

                # uncomment this code if you want to change ucs to bfs
                # newNode[2].append(int(currState[2][0])
                #                   + 1)
                fringe.push(newNode, newNode[2][0])
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initialize the search tree using initial state of the problem
    fringe = util.PriorityQueue()
    currState = [[]]
    currState.append([])
    currState.append([])
    currState[0].append(problem.getStartState())
    currState[2].append(0)

    fringe.push(item=currState, priority=currState[2][0])
    fringeNodes = []
    while True:
        if fringe.isEmpty():
            return None
        currState = fringe.pop()
        if currState[0][0] in fringeNodes:
            continue
        fringeNodes.append(currState[0][0])
        if problem.isGoalState(currState[0][0]):
            return currState[1]
        successors = problem.getSuccessors(currState[0][0])
        for i in range(len(successors)):
            if not (successors[i][0] in fringeNodes):
                newNode = [[]]
                newNode.append([])
                newNode.append([])
                newNode[0].append(successors[i][0])
                newNode[1] = list(currState[1]).copy()
                newNode[1].append(successors[i][1])
                newNode[2].append(int(currState[2][0]) + successors[i][2])
                fringe.push(newNode, newNode[2][0] + heuristic(successors[i][0], problem))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
