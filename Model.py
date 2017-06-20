# -*- coding: utf-8 -*-
"""
Haven't really done much
"""

'''
Forces
    Person-person
    Person-wall
    Person-robot
    Person-goal
        Tends to move in the direction they are facing
        Move toward where they want to go 
            Other side of the room in Sharks and Minnows, abbreviated S&M henceforth
    
'''

import math
import numpy as np

######################################################################
#                                                                    #
#                             AGENTS                                 #
#                                                                    #
######################################################################

class Agent:
    '''
    The class that defines the agents
    
    Things to add
        Initial position, orientation angle, and velocity
        Variable radius and mass
        
    '''
    
    number = 0
    radius = 0.25
    mass = 60
    
    def __init__(self, number, position, velocity):
        self.number = number
        self.position = position
        self.velocity = velocity

        Agent.number += 1
    
    def xVals(self):
        '''
        x position and velocity display
        '''
        xPos = 'x position: ' + str(self.position[0])
        xVel =  'x velocity: ' + str(self.velocity[0])
        xVals = xPos + '\n' + xVel
        return xVals
    
    def yVals(self):
        '''
        y position and velocity display
        '''
        yPos = 'y position: ' + str(self.position[1])
        yVel =  'y velocity: ' + str(self.velocity[1])
        yVals = yPos + '\n' + yVel
        return yVals

def agentToAgentDistance(agentI,agentJ):
    '''
    This function is meant to calculate the total distance between two
    different agents.
    '''
    xDif = abs(agentI.position[0] - agentJ.position[0])
    yDif = abs(agentI.position[1] - agentJ.position[1])
    totDif = math.sqrt(xDif**2 + yDif**2)
    return totDif

def agentToAgentVector(agentI, agentJ):
    '''
    This function calculates the unit vector pointing from agent i to agent j
    '''
    dist = agentToAgentDistance(agentI, agentJ)
    xVect = (agentI.position[0]-agentJ.position[0])/dist
    yVect = (agentI.position[1]-agentJ.position[1])/dist
    unitVect = [xVect, yVect]
    return unitVect
    

######################################################################
#                                                                    #
#                          OBSTACLES                                 #
#                                                                    #
######################################################################


class Obstacle:
    
    number = 0
    
    def __init__(self, xOne, yOne, xTwo, yTwo):
        '''
        This creates a rectangular obstacle based on the coordinates
        (xOne, yOne) and (xTwo, yTwo)
        
        It creates two arrays, one for x values and one for y values,
        start at the lower value and go to the higher value incrementing
        by a set value.
        
        I think that this is the finite element method but I haven't taken 
        that class yet so...
        
        '''
        
        # We might want to add something to reject construction if points one
        # and two are the same point
        
        self.xMax = max(xOne, xTwo)
        self.xMin = min(xOne, xTwo)
        self.yMax = max(yOne, yTwo)
        self.yMin = min(yOne, yTwo)
        
        self.pointOne = [self.xMax, self.yMax]
        self.pointTwo = [self.xMin, self.yMin]
        
        # Could be used to determine the interval size for the arrays below,
        # cause we might not need 100 points between 1 and 1.1
        self.xRange = self.xMax - self.xMin
        self.yRange = self.yMax - self.yMin
        
        self.points = 100
        
        # This is the main part right now, it's an array containing 100 points,
        # the min and max values with 98 evenly spaced points between them
        # IDEALLY we get rid of everything but the borders cause there will
        # never be a point when the closest point to someone is inside of the
        # obstacle unless they are forced inside
        self.xVals = np.linspace(self.xMin, self.xMax, num=self.points)
        self.yVals = np.linspace(self.yMin, self.yMax, num=self.points)
        
        Obstacle.number += 1
        
    @classmethod
    def circularObstacle(self, x, y, radius):
        '''
        This is going to be an alternate constructor that can create circular
        objects at the given point (x, y) of a given radius
        We could also add angles it goes through instead of 360 always, and
        add the option for an inner radius
        '''
        pass
        
    
        
######################################################################
#                                                                    #
#                    HEADED SOCIAL FORCE MODEL                       #
#                                                                    #
######################################################################

# These are constants for calculations
tau = 0.5           # Seconds
A = 2*(10**3)       # Newtons
B = 0.08            # Meters
k1 = 1.2*(10**5)    # kg/(s**2)
k2 = 2.4*(10**5)    # kg/(m*s)

def agentToAgentForce(agentI, agentJ):
    '''
    This function calculates the repulsive force of agent J on agent I
    
    The compressive and friction terms only come into play when the distance
    between the two agents is less than the two agent's radius
    
    All of these come from the Headed Social Force Model paper
    '''
    
    radiusSum = agentI.radius + agentJ.radius
    dist = agentToAgentDistance(agentI, agentJ)
    unitVect = np.array(agentToAgentVector(agentI, agentJ))
    perpUnitVect = np.array([-unitVect[1], unitVect[0]])
    velDif = np.dot(np.array([agentJ.velocity[0] - agentI.velocity[0],
                       agentJ.velocity[1] - agentI.velocity[1]]), perpUnitVect)
    
    repulsiveTerm = A * math.exp((radiusSum - dist)/B) * unitVect
    
    compressiveTerm = (k1 * max(radiusSum - dist, 0)) * unitVect
    
    frictionTerm = (k2 * max(radiusSum - dist, 0)) * velDif * perpUnitVect

    summedForce = repulsiveTerm + compressiveTerm + frictionTerm
    
    return summedForce

def agentToObstacleForce(agent, obstacle):
    '''
    This function calculates the force between an agent and an obstacle
    '''
    
    # Need to calculate the closest point between the agent and the wall
    xi = 0
    yi = 0
    currentClosestX = 1000
    currentClosestY = 1000
    while (xi < obstacle.points):
        # Run through the array of x points for the obstacle, obstacle.xVals, and see
        # which gives the closest point, closestObstacleX
        xVal = abs(obstacle.xVals[xi] - agent.position[0])
        if xVal < currentClosestX:
            currentClosestX = xVal
            closestObstacleX = obstacle.xVals[xi]
        xi += 1
    
    while (yi < obstacle.points):
        # Same process as above
        yVal = abs(obstacle.xVals[yi] - agent.position[1])
        if yVal < currentClosestY:
            currentClosestY = yVal
            closestObstacleY = obstacle.xVals[yi]
        yi += 1
    

    

    return [closestObstacleX, closestObstacleY]

agentOne = Agent(1, [1,7], [1,1])
agentTwo = Agent(2, [2,2], [3,2])

force = agentToAgentForce(agentOne,agentTwo)

wallOne = Obstacle(2, 2, 5, 5)

obstacleForce = agentToObstacle(agentOne,wallOne)

print obstacleForce



