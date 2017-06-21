# -*- coding: utf-8 -*-
'''
Forces
    Person-person
    Person-wall
    Person-robot
    Person-goal
        Tends to move in the direction they are facing
        Move toward where they want to go 
            Other side of the room in Sharks and Minnows, abbreviated S&M henceforth

Units
    Try to do everything with the standard SI units for easy conversions
    meters
    meters/second
    kg
    seconds
    newtons
    etc
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
    
    def __init__(self, number, position, velocity, headingVector):
        self.number = number
        self.position = position
        self.velocity = velocity
        self.headingVector = headingVector
        self.angle = math.radians(headingVector[0])
        self.angularVelocity = headingVector[1]
        
        self.momentOfInertia = 0.5 * self.mass * self.radius**2
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
        
        # Not necessary right now, but if we need to graph it this might be useful
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
        
def agentToObstacleDistance(agent, closestX, closestY):
    '''
    This function is calculates the total distance between the agent and obstacle
    '''
    xDif = abs(agent.position[0] - closestX)
    yDif = abs(agent.position[1] - closestY)
    totDif = math.sqrt(xDif**2 + yDif**2)
    return totDif
    
def agentToObstacleVector(agent, closestX, closestY):
    dist = agentToObstacleDistance(agent, closestX, closestY)
    xVect = (agent.position[0]-closestX)/dist
    yVect = (agent.position[1]-closestY)/dist
    unitVect = [xVect, yVect]
    return unitVect
    
######################################################################
#                                                                    #
#                    HEADED SOCIAL FORCE MODEL                       #
#                                                                    #
######################################################################

# These are constants for calculations, taken from the headed social force model paper
tau = 0.5           # Seconds
A = 2*(10**3)       # Newtons
B = 0.08            # Meters
k1 = 1.2*(10**5)    # kg/(s**2)
k2 = 2.4*(10**5)    # kg/(m*s)

######################################################################
#                       SOCIAL FORCE MODEL                           #
######################################################################

def agentToGoalForce(agentI, goalPoint):
    '''
    This function calculates the force between the agent and their goal point
    
    It calculates the difference between the person and their ideal velocity
    '''
    # To start, find the vector that the person would like to be moving at
    # i.e. the vector in the direction of their point at their max speed
    goalVectX = goalPoint[0] - agentI.position[0]
    goalVectY = goalPoint[1] - agentI.position[1]
    dist = math.sqrt(goalVectX**2 + goalVectY**2)
    goalSpeed = 1.388 # This is the average speed of a human in meters/second
    goalVelX = goalVectX * goalSpeed / dist
    goalVelY = goalVectY * goalSpeed / dist
    goalVelocity = [goalVelX, goalVelY]
    
    timeIncrement = 1 # Seconds
    mass = agentI.mass
    
    forceX = mass * (goalVelocity[0] - agentI.velocity[0]) / timeIncrement
    forceY = mass * (goalVelocity[1] - agentI.velocity[1]) / timeIncrement
    
    forceVect = np.array([[forceX], [forceY]])
    return forceVect

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
    
    return np.array([[summedForce[0]],[summedForce[1]]])

def agentToObstacleForce(agent, obstacle):
    '''
    This function calculates the force between an agent and an obstacle.
    
    First it calculates the closest point to the agent on the obstacle, there are
    three cases for a rectangular obstacle
        The agent is parallel to a wall with constant y
            Chooses the agent's x coordinate and the closest wall's y coordinate
        The agent is parallel to a wall with constant x
            Chooses the agent's y coordinate and the closest wall's x coordinate
        The agent is closest to one of the corners
            Chooses the closest corner
    ''' 
    
    if agent.position[0] in range(obstacle.xMin, obstacle.xMax):
        '''
        If the agent is parallel to a wall with constant y then choose the point that
        matches the x coordinate of the agent and the closest wall's y coordinate
        '''
        closestX = agent.position[0]
        
        if abs(agent.position[1]-obstacle.yMin) < abs(agent.position[1]-obstacle.yMax):
            closestY = obstacle.yMin
        else:
            closestY = obstacle.yMax
            
    elif agent.position[1] in range(obstacle.yMin, obstacle.yMax):
        '''
        If the agent is parallel to a y wall then choose the point that
        matches the y coordinate of the agent and the closest wall's x coordinate
        '''
        if abs(agent.position[0]-obstacle.xMin) < abs(agent.position[0]-obstacle.xMax):
            closestX = obstacle.xMin
        else:
            closestX = obstacle.xMax
            
        closestY = agent.position[1]
        
    else:
        # Choose a corner if the agent is not within either x or y ranges
        if abs(agent.position[0]-obstacle.xMin) < abs(agent.position[0]-obstacle.xMax):
            closestX = obstacle.xMin
        else:
            closestX = obstacle.xMax
        
        if abs(agent.position[1]-obstacle.yMin) < abs(agent.position[1]-obstacle.yMax):
            closestY = obstacle.yMin
        else:
            closestY = obstacle.yMax

    # This is all taken directly from the headed social force model paper
    radiusSum = agent.radius
    dist = agentToObstacleDistance(agent, closestX, closestY)
    unitVect = np.array(agentToObstacleVector(agent, closestX, closestY))
    perpUnitVect = np.array([-unitVect[1], unitVect[0]])
    velDif = np.dot(np.array([0 - agent.velocity[0], 0 - agent.velocity[1]]), perpUnitVect)
    
    repulsiveTerm = A * math.exp((radiusSum - dist)/B) * unitVect
    
    compressiveTerm = (k1 * max(radiusSum - dist, 0)) * unitVect
    
    frictionTerm = (k2 * max(radiusSum - dist, 0)) * velDif * perpUnitVect

    summedForce = repulsiveTerm + compressiveTerm + frictionTerm
    
    return np.array([[summedForce[0]],[summedForce[1]]])

######################################################################
#                     CONTROL INPUT CALCULATION                      #
######################################################################

kf = 1
k0 = 0.3
kd = 5
alpha = 3
kLambda = 0.02

def rotationMatrix(angle):
    '''
    Calculates the rotation matrix
    '''
    R = np.array([[math.cos(angle), -math.sin(angle)],
                   [math.sin(angle),math.cos(angle)]])
    return R

def inverseRotationMatrix(angle):
    '''
    Calculates the rotation matrix
    '''
    R = np.array([[math.cos(angle), math.sin(angle)],
                   [-math.sin(angle),math.cos(angle)]])
    return R

def bodyVector(agent):
    '''
    We need to calculate the the velocity vector in the body frame, aka the
    
    '''
    theta = agent.angle
    velocityB = np.matmul(inverseRotationMatrix(theta), agent.velocity)
    return velocityB

def controlInputB(agent, totalForce):
    '''
    A function that calculates one of the velocity inputs for the human
    locomotion model.  It represents the forces acting in the forward direction
    and the sideward direction of the agent.
    '''
    # I SHOULD GO OVER THIS AGAIN AND MAKE SURE THAT IT'S RIGHT
    KB = np.array([[kf, 0], [0, k0]])
    kdVect = np.array([[0], [kd]])
    rotationM = np.transpose(rotationMatrix(agent.angle))
    vi = bodyVector(agent)
    
    
    uB = np.matmul(np.matmul(KB, rotationM), totalForce) - (vi[1] * kdVect)
    
    return uB

def controlInputTheta(agent, goalForce):
    '''
    The theta input represents the torque about the vertical axis that
    drives the angle change of the agent.
    '''
    theta = agent.angle
    omega = agent.angularVelocity
    goalForceMagnitude = math.sqrt(goalForce[0]**2 + goalForce[1]**2)
    goalTheta = math.atan(goalForce[1] / goalForce[0])
    
    kTheta = agent.momentOfInertia * kLambda * goalForceMagnitude
    kOmega = agent.momentOfInertia * (1 + alpha) * math.sqrt(kLambda * goalForceMagnitude / alpha)
    
    uTheta = -kTheta * (theta - goalTheta) - kOmega * omega
    return uTheta



agentOne = Agent(1, [5, 3.2], [.25,1], [45,1])
agentTwo = Agent(2, [0, 0], [1,1], [1,1])

goalPoint = [10, 10]

wallOne = Obstacle(4,4,6,6)

agentToGoalForce(agentOne, goalPoint)

goalForce = agentToGoalForce(agentOne, goalPoint)
agentForce = agentToAgentForce(agentOne, agentTwo)
obstacleForce = agentToObstacleForce(agentOne,wallOne)

totalForce = goalForce + agentForce + obstacleForce

input1 = controlInputB(agentOne, totalForce)
input2 = controlInputTheta(agentOne, goalForce)
print bodyVector(agentOne)

print input2