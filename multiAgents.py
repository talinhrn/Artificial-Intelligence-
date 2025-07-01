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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        # Get the current score of the successor state
        score = successorGameState.getScore()
        
        # Food
        foodList = newFood.asList()
        closestFoodDist = float('inf')  
        for food in foodList:
            dist = util.manhattanDistance(newPos, food)  
            if dist < closestFoodDist:  
                    closestFoodDist = dist  
                    
        if closestFoodDist != float('inf'):  
            score += 1 / closestFoodDist  

        # Avoid ghosts
        ghostDist = []
        closestGhost  = float('inf')  
        for ghostState in newGhostStates:
            ghostDist = util.manhattanDistance(newPos, ghostState.getPosition())
            if ghostDist < closestGhost:
                closestGhost = ghostDist
            if closestGhost != float('inf'): 
                if closestGhost < 2:
                   score -= 1
                else:
                    score += 1 / closestGhost

        # For stopping
        if action == Directions.STOP:
            score -= 1

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        util.raiseNotDefined()

        """
        "*** YOUR CODE HERE ***"
                    
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # Pacman
            if agentIndex == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))
            else:  
                nextAgent = agentIndex + 1  
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0  #Pacman
                    depth += 1  
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action)) for action in gameState.getLegalActions(agentIndex))

        # Minimax search for Pacman 
        legalMoves = gameState.getLegalActions(0)
        bestAction = []
        maxScore = float('-inf')

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)  
            actionScore = minimax(1, 0, successor)  
            if actionScore > maxScore:
                maxScore = actionScore  
                bestAction = action 

        return bestAction
        
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        value, action = self.alphaBeta(gameState, 0, float('-inf'), float('+inf'))
        return action

    def alphaBeta(self, gameState, depth, alpha, beta):
        """
        Alpha-beta pruning algorithm with value updates for Pacman and ghosts.
        """
        numAgents = gameState.getNumAgents()
        agentIndex = depth % numAgents  # Current agent index

        # Base case: terminal state or depth reached
        if depth // numAgents == self.depth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), []

        legalActions = gameState.getLegalActions(agentIndex)
        if not legalActions:
            return self.evaluationFunction(gameState), []

        # Pacman
        if agentIndex == 0:  
            v_a = float('-inf')
            bestAction = []
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                successorValue, nextAction = self.alphaBeta(successor, depth + 1, alpha, beta)
                if successorValue > v_a:
                    v_a = successorValue
                    bestAction = action
                if v_a > beta:
                    return v_a, bestAction
                alpha = max(alpha, v_a)
            return v_a, bestAction
       
        # Ghost
        else:  
            v_b = float('+inf')
            bestAction = []
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                successorValue, nextAction = self.alphaBeta(successor, depth + 1, alpha, beta)
                if successorValue < v_b:
                    v_b = successorValue
                    bestAction = action
                if v_b < alpha:
                    return v_b, bestAction
                beta = min(beta, v_b)
            return v_b, bestAction
    
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (Question 4).
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.

        All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(agent, depth, gameState):
            """
            Recursive function to perform expectimax search.
            """
            
            legalActions = gameState.getLegalActions(agent)

            # Base case
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if not legalActions:  
                return self.evaluationFunction(gameState)

            # Pacman
            if agent == 0:  
                bestValue = float('-inf')  
                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    value = expectimax(1, depth, successor)  
                    bestValue = max(bestValue, value)  
                return bestValue

            # Ghost
            else:
                nextAgent = agent + 1  
                if nextAgent == gameState.getNumAgents():  
                    nextAgent = 0
                if nextAgent == 0:  
                    depth = depth + 1

                # Expected value
                expectedValue = 0
                for action in legalActions:
                    successor = gameState.generateSuccessor(agent, action)
                    value = expectimax(nextAgent, depth, successor)  
                    expectedValue += value  
                return expectedValue / len(legalActions)  

        # Expectimax
        bestAction = []
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):  
            successor = gameState.generateSuccessor(0, action)
            successorValue = expectimax(1, 0, successor) 
            if bestValue < successorValue:
                bestValue = successorValue
                bestAction = action

        return bestAction
    

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    
    - This function balances multiple factors to evaluate the desirability of a state:
    1. Food Distance: Rewards states where Pacman is closer to food.
    2. Ghost Proximity: Penalizes states where Pacman is too close to ghosts.
    3. Capsule Distance: Rewards states where Pacman is closer to capsules (power pellets).
    4. Game Score: Incorporates the game score to prioritize higher scoring states.
    5. Remaining Food: Penalizes states with more remaining food.
    
    
    """
    "*** YOUR CODE HERE ***"
    
    position = currentGameState.getPacmanPosition()

    # Initialize 
    foodDistanceScore = 0
    ghostProximityPenalty = 0
    capsuleScore = 0
    minDist = -1

    # Food Distance
    food = currentGameState.getFood()
    foodList = food.asList()
    for food in food.asList():
        dist = manhattanDistance(position, food)
        if dist <= minDist or minDist == -1:
            minDist = dist
    
    # Ghost 
    ghosts = currentGameState.getGhostStates()
    for ghost in ghosts:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(position, ghostPos)
        if ghost.scaredTimer == 0:  
            if ghostDist > 0:
                ghostProximityPenalty += -10.0 / ghostDist 
        else:  
            ghostProximityPenalty += 10.0 / (ghostDist + 1)

    # Capsule Bonus
    capsules = currentGameState.getCapsules()    
    if capsules:
        nearCapsule = min([manhattanDistance(position, capsule) for capsule in capsules])
        capsuleScore = 10.0 / (nearCapsule + 1)  

    
    foodPenalty = -len(foodList) * 4.0
    foodScore = 5.0 * foodDistanceScore  
    ghostPenalty = ghostProximityPenalty 

    # Total score
    score = currentGameState.getScore()
    totScore = score + capsuleScore + foodPenalty + foodScore + ghostPenalty

    return totScore

# Abbreviation
better = betterEvaluationFunction
