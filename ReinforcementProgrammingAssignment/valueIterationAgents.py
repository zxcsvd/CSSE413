# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent
import math

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
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
    self.delta = 0
    while(self.iterations > 0):
#         self.delta = 0
        batchValues = util.Counter()
        for state in mdp.getStates():  
            maxM = -10000
                   
            if mdp.isTerminal(state):
                continue 
            for action in mdp.getPossibleActions(state):
                statesProbs = mdp.getTransitionStatesAndProbs(state, action)
                sumU = 0
                Rs = 0
                for stateProb in statesProbs:
#                     if stateProb[0] == 'TERMINAL_STATE':
#                         continue
                    sumU = sumU + self.values[stateProb[0]]*stateProb[1]
                    Rs = Rs + mdp.getReward(state, action, stateProb[0]) * stateProb[1]
#                 if sumU > maxM:
#                     maxM = sumU   
                v = Rs + sumU * discount
                if (v > maxM):
                    maxM = v
            batchValues[state] = maxM
        self.values = batchValues
        self.iterations = self.iterations - 1       
    self.policy = {}
    for state in mdp.getStates():
        if mdp.isTerminal(state):
            self.policy[state] = None
            continue
        QValues = []
        for action in mdp.getPossibleActions(state):
            QValues.append(self.getQValue(state, action))
            self.policy[state] = mdp.getPossibleActions(state)[QValues.index(max (QValues))]
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    
    "*** YOUR CODE HERE ***"
    statesProbs = self.mdp.getTransitionStatesAndProbs(state, action)
    sumU = 0
    Rs = 0
    for stateProb in statesProbs:
        if stateProb == 'TERMINAL_STATE':
            continue
        sumU = sumU + self.values[stateProb[0]]*stateProb[1]
        Rs = self.mdp.getReward(state, action, stateProb[0])*stateProb[1]    
    return Rs + sumU * self.discount

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    return self.policy[state]
#     util.raiseNotDefined()

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
