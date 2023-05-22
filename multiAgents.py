# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == "Stop":
            return -999999
        if successorGameState.isLose():
            return -999999
        if successorGameState.isWin():
            return 999999

        minDistanceToFood = 999999
        for food in newFood:
            if manhattanDistance(food, newPos) < minDistanceToFood:
                minDistanceToFood = manhattanDistance(food, newPos)

        for ghost in newGhostStates:
            if ghost.getPosition() == newPos and newScaredTimes == 0:
                return -999999

        currFood = currentGameState.getFood().asList()
        if newPos in currFood:
            return 999999

        return -1 * minDistanceToFood + successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        nGhosts = gameState.getNumAgents() - 1

        def minValue(state, ghostNumber, depth):
            v = 999999
            CorrectActions = state.getLegalActions(ghostNumber)

            if len(CorrectActions) == 0:
                return self.evaluationFunction(state)

            for action in CorrectActions:
                thisGhostNextState = state.generateSuccessor(ghostNumber, action)
                if ghostNumber == nGhosts:
                    v = min(v, maxValue(thisGhostNextState, depth))
                else:
                    v = min(v, minValue(thisGhostNextState, ghostNumber + 1, depth))
            return v

        def maxValue(state, depth):
            v = -999999
            if (depth == self.depth):
                return self.evaluationFunction(state)

            CorrectActions = state.getLegalActions(0)
            if len(CorrectActions) == 0:
                return self.evaluationFunction(state)

            for action in CorrectActions:
                pacmanNextState = state.generateSuccessor(0, action)
                v = max(v, minValue(pacmanNextState, 1, depth + 1))
            return v

        CorrectActions = gameState.getLegalActions(0)
        bestEffort = -999999
        for action in CorrectActions:
            newEffort = minValue(gameState.generateSuccessor(0, action), 1, 1)
            if newEffort > bestEffort:
                bestEffort = newEffort
                result = action
        return result
        # util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        nGhosts = gameState.getNumAgents() - 1

        def minValue(state, ghostNumber, depth, alpha, beta):
            v = 999999
            correctActions = state.getLegalActions(ghostNumber)
            if len(correctActions) == 0:
                return self.evaluationFunction(state)

            for action in correctActions:
                thisGhostNextState = state.generateSuccessor(ghostNumber, action)
                if ghostNumber == nGhosts:
                    v = min(v, maxValue(thisGhostNextState, depth, alpha, beta))
                else:
                    v = min(v, minValue(thisGhostNextState, ghostNumber + 1, depth, alpha, beta))

                if v < alpha: return v
                beta = min(beta, v)
            return v

        def maxValue(state, depth, alpha, beta):
            v = -999999
            if (depth == self.depth):
                return self.evaluationFunction(state)

            correctActions = state.getLegalActions(0)  # Pacman is agent 0
            # no correct actions, we have either lost or won the game
            if len(correctActions) == 0:
                return self.evaluationFunction(state)

            # will return illegal action otherwise
            if depth == 0: result = correctActions[0]
            for action in correctActions:
                pacmanNextState = state.generateSuccessor(0, action)
                depthCondition = minValue(pacmanNextState, 1, depth + 1, alpha, beta)
                if (depthCondition > v):
                    if depth == 0: result = action
                    v = depthCondition
                if v > beta: return v
                alpha = max(alpha, v)

            # will return illegal action otherwise
            if depth == 0: return result
            return v

        ## same as q2
        return maxValue(gameState, 0, -999999, 999999)
        # util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        nGhosts = gameState.getNumAgents() - 1

        def maxValue(state, depth):
            v = -999999
            if (depth == self.depth):
                return self.evaluationFunction(state)

            correctActions = state.getLegalActions(0)  # Pacman is agent 0
            # no correct actions, we have either lost or won the game
            if len(correctActions) == 0:
                return self.evaluationFunction(state)

            for action in correctActions:
                pacmanNextState = state.generateSuccessor(0, action)
                v = max(v, expValue(pacmanNextState, 1, depth + 1))
            return v

        def expValue(state, ghostNumber, depth):
            v = 0
            correctActions = state.getLegalActions(ghostNumber)
            # no correct actions, we have either lost or won the game
            if len(correctActions) == 0:
                return self.evaluationFunction(state)

            for action in correctActions:
                thisGhostNextState = state.generateSuccessor(ghostNumber, action)
                p = 1.0 / len(correctActions)
                # all ghosts have moved, now it's pacman's turn
                if ghostNumber == nGhosts:
                    v += p * maxValue(thisGhostNextState, depth)
                else:  # next ghost's turn
                    v += p * expValue(thisGhostNextState, ghostNumber + 1, depth)
            return v

        # Pacman starts the game
        CorrectActions = gameState.getLegalActions(0)
        bestEffort = -999999
        for action in CorrectActions:
            # start of the ricursive calls
            newEffort = expValue(gameState.generateSuccessor(0, action), 1, 1)
            if newEffort > bestEffort:
                bestEffort = newEffort
                result = action
        return result
        ## util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"

    sumGhostsDistances = 0
    for position in ghostPositions:
        sumGhostsDistances += manhattanDistance(pacmanPosition, position)

    sumFoodsDistances = 0
    foodList = foods.asList()
    for food in foodList:
        sumFoodsDistances += manhattanDistance(pacmanPosition, food)

    foodsEaten = len(foods.asList(False))

    return currentGameState.getScore() + foodsEaten + sumGhostsDistances \
           - sumFoodsDistances + sum(scaredTimers)


# Abbreviation
better = betterEvaluationFunction
