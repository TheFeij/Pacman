# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from numpy import Infinity
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            myDictionary = util.Counter()
            for state in self.mdp.getStates():
                possibleActions = self.mdp.getPossibleActions(state)
                bestEffort = -Infinity
                for action in possibleActions:
                    newEffort = self.computeQValueFromValues(state, action)
                    if newEffort > bestEffort:
                        bestEffort = newEffort
                if bestEffort != -Infinity:
                    myDictionary[state] = bestEffort
            for state in self.mdp.getStates():
                self.values[state] = myDictionary[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        result = 0
        ## (nextState, prob) pairs (from mdp)
        nextStatesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        ## index 0 is next state and index 1 is the probability
        for nextStateProb in nextStatesProbs:
            T = nextStateProb[1]
            R = self.mdp.getReward(state, action, nextStateProb[0])
            ## V_k(S_prime) = self.values[nextStateProb[0]]
            gammaV = self.discount * self.values[nextStateProb[0]]
            ## Q(s,a) = sigma(T*(R+gammaV))
            result += T * (R + gammaV)
        return result

        ##util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        ## last two lines from above
        if self.mdp.isTerminal(state):
            return None
        
        possibleActions = self.mdp.getPossibleActions(state)
        bestEffort = -Infinity
        for action in possibleActions:
            ## P(s) = argmax_a(Q(s,a))
            newEffort = self.computeQValueFromValues(state, action)
            if newEffort > bestEffort:
                bestEffort = newEffort
                result = action
        return result

        ##util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        numberOfAllStates = len(allStates)
        for i in range(self.iterations):
            stateIndexInThisCycle = i % numberOfAllStates
            ## performing cyclic value iteration
            stateInThisCycle = allStates[stateIndexInThisCycle]
            if self.mdp.isTerminal(stateInThisCycle) == False:
                possibleActions = self.mdp.getPossibleActions(stateInThisCycle)
                bestEffort = -Infinity
                for action in possibleActions:
                    ## using computeQValueFromValues by heritage (getQValue) 
                    newEffort = self.getQValue(stateInThisCycle, action)
                    if newEffort > bestEffort:
                        bestEffort = newEffort
                self.values[stateInThisCycle] = bestEffort


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        ## using sets so as not to have duplicate states
        Predecessors = {}
        # part one
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state) == False:
                possibleActions = self.mdp.getPossibleActions(state)
                for action in possibleActions:
                    statesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for stateAndProb in statesAndProbs:
                        if stateAndProb[0] not in Predecessors:
                            Predecessors[stateAndProb[0]] = {state}
                        else:
                            Predecessors[stateAndProb[0]].add(state)

        ## part two
        priQueue = util.PriorityQueue()
        ## part three
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == False:
                possibleActions = self.mdp.getPossibleActions(s)
                bestEffort = -Infinity
                for action in possibleActions:
                    newEffort = self.computeQValueFromValues(s, action)
                    if newEffort > bestEffort:
                        bestEffort = newEffort
                diff = self.values[s] - bestEffort
                if diff < 0:
                    diff = diff * (-1)
                priQueue.update(s, -diff)
                
        ## part four
        for i in range(self.iterations):
            if priQueue.isEmpty():
                break
            s = priQueue.pop()
            if self.mdp.isTerminal(s) == False:
                possibleActions = self.mdp.getPossibleActions(s)
                bestEffort = -Infinity
                for action in possibleActions:
                    newEffort = self.computeQValueFromValues(s, action)
                    if newEffort > bestEffort:
                        bestEffort = newEffort
                self.values[s] = bestEffort
            for p in Predecessors[s]:
                if self.mdp.isTerminal(p) == False:
                    possibleActions = self.mdp.getPossibleActions(p)
                    bestEffort = -Infinity
                    for action in possibleActions:
                        newEffort = self.computeQValueFromValues(p, action)
                        if newEffort > bestEffort:
                            bestEffort = newEffort
                    diff = self.values[p] - bestEffort
                    if diff < 0:
                        diff = diff * (-1)
                    if diff > self.theta:
                        priQueue.update(p, -diff)

