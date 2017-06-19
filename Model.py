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

class Agent:
    '''
    The class that defines the agents
    '''
    
    radius = 0.25
    mass = 60
    
    def __init__(self, number, position, velocity):
        self.number = number
        self.position = position
        self.velocity = velocity
        
    
    def xVals(self):
        xPos = 'x position: ' + str(self.position[0])
        xVel =  'x velocity: ' + str(self.velocity[0])
        xVals = xPos + '\n' + xVel
        return xVals
    
    def yVals(self):
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
    This function takes in two different agents and then calculates the
    force between them
    '''
    radiusSum = agentI.radius + agentJ.radius
    dist = agentToAgentDistance(agentI, agentJ)
    velDif = np.array([agentJ.velocity[0] - agentI.velocity[0], agentJ.velocity[1] - agentI.velocity[1]])
    unitVect = np.array(agentToAgentVector(agentI, agentJ))
    perpUnitVect = np.array([-unitVect[1], unitVect[0]])
    
    repulsiveTerm = A * math.exp((radiusSum - dist)/B) * unitVect
    
    compressiveTerm = (k1 * max(radiusSum - dist, 0)) * unitVect
    
    frictionTerm = (k2 * max(radiusSum - dist, 0))
    
    print velDif
    print compressiveTerm
    print repulsiveTerm
    
        
agentOne = Agent(1, [1,1.1], [2,2])
agentTwo = Agent(2, [1.1,1], [2,2])
agentThree = Agent(2, [20,25], [2,-1])

agentToAgentForce(agentOne,agentTwo)

#print agentOne.mass
#print agentTwo.position
#print agentThree.position

#distance12 = agentToAgentDistance(agentOne,agentTwo)
#vector12 = agentToAgentVector(agentOne,agentTwo)
#
#distance13 = agentToAgentDistance(agentOne,agentThree)
#vector13 = agentToAgentVector(agentOne,agentThree)
#
#print '\nThe distance between 1 and 2 is: ' + str(distance12)
#print '\nThe unit vector between 1 and 2 is:\n' + str(vector12)
#
#print '\nThe distance between 1 and 3 is: ' + str(distance13)
#print '\nThe unit vector between 1 and 3 is:\n' + str(vector13)
