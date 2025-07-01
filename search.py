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
from game import Directions
from typing import List


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




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    
    # Use stack as structure for DFS
    node = util.Stack()
    visited = []
    node.push((problem.getStartState(), []))  

    while not node.isEmpty():
        current_state, path = node.pop()
        
        if problem.isGoalState(current_state):
            return path
        
        if current_state not in visited:
            visited.append(current_state)
            
            for next_state, next_action, _ in problem.getSuccessors(current_state):
                if next_state not in visited:
                    node.push((next_state, path + [next_action]))
    
    return []

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    
    # Initialize the frontier and paths queues
    is_visited = []
    actions = []
    queue = util.Queue()
    paths = util.Queue()
    queue.push(problem.getStartState())
    paths.push([])

    while queue:
        current_state = queue.pop()
        current_path = paths.pop()
        if current_state in is_visited:
            continue
        is_visited.append(current_state)
        
        if problem.isGoalState(current_state):
            actions = current_path
            break
   
        for next_state, next_action, _ in problem.getSuccessors(current_state):
            if next_state not in is_visited:
                queue.push(next_state)  
                paths.push(current_path + [next_action])  

    return actions  

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    # Priority Queue to hold nodes with their associated cost
    
    start_state = problem.getStartState()
    queue = util.PriorityQueue()
    queue.push((start_state, []), 0)
    visited = set()
   
    while queue:
       current_state, path = queue.pop()
       
       # Check 
       if problem.isGoalState(current_state):
           return path
       
       if current_state not in visited:
           visited.add(current_state)  # Mark current state as visited
           
           # Explore all successors 
           for next_state, next_action, step_cost in problem.getSuccessors(current_state):
               if next_state not in visited:  
                   new_path = path + [next_action]
                   new_cost = problem.getCostOfActions(new_path)  
                   queue.push((next_state, new_path), new_cost)

    return []

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # similar to above functions
    queue = util.PriorityQueue()
    queue.push((problem.getStartState(), []), 0)
    visited = []

    while queue:
        current_state, path = queue.pop()

        if problem.isGoalState(current_state): 
            return path
        elif current_state not in visited:  
            visited.append(current_state)
            for successor in problem.getSuccessors(current_state):  
                next_state, next_action, step_cost = successor
                if next_state not in visited:
                    new_path = path + [next_action]
                    new_cost = problem.getCostOfActions(new_path) + heuristic(next_state, problem)
                   
                    queue.push((next_state, new_path), new_cost)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch



