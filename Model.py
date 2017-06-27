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
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    agentList = []
    number = 0
    radius = 0.25
    mass = 60
    
    def __init__(self, position, velocity, headingVector, goalPoint):
        self.number = Agent.number
        self.position = position
        self.velocity = velocity
        self.headingVector = headingVector
        self.angle = math.radians(headingVector[0])
        self.angularVelocity = headingVector[1]
        self.goalPoint = goalPoint
        
        self.newPosition = [0, 0]
        self.newVelocity = [0, 0]
        
        self.momentOfInertia = 0.5 * self.mass * self.radius**2
        Agent.agentList.append(self)
        Agent.number += 1
    

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
    if dist == 0:
        return 0
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
    obstacleList = []
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
        Obstacle.obstacleList.append(self)
        
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
    if dist == 0:
        return np.zeros(2)
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
timeIncrement = 1   # Seconds

######################################################################
#                       SOCIAL FORCE MODEL                           #
######################################################################

def agentToGoalForce(agent):
    '''
    This function calculates the force between the agent and their goal point
    
    It calculates the difference between the person and their ideal velocity
    '''
    # To start, find the vector that the person would like to be moving at
    # i.e. the vector in the direction of their point at their max speed
    goalVectX = agent.goalPoint[0] - agent.position[0]
    goalVectY = agent.goalPoint[1] - agent.position[1]
    dist = math.sqrt(goalVectX**2 + goalVectY**2)
    goalSpeed = 1.388 # This is the average speed of a human in meters/second
    goalVelX = goalVectX * goalSpeed / dist
    goalVelY = goalVectY * goalSpeed / dist
    goalVelocity = [goalVelX, goalVelY]
    

    mass = agent.mass
    
    forceX = mass * (goalVelocity[0] - agent.velocity[0]) / timeIncrement
    forceY = mass * (goalVelocity[1] - agent.velocity[1]) / timeIncrement
    
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
    perpUnitVect.reshape(2, 1)
    velDifArray = np.array([agentJ.velocity[0] - agentI.velocity[0],
                       agentJ.velocity[1] - agentI.velocity[1]])
    
    repulsiveTerm = A * math.exp((radiusSum - dist)/B) * unitVect
    
    compressiveTerm = (k1 * max(radiusSum - dist, 0)) * unitVect
    
    if velDifArray[0] == 0 and velDifArray[1] == 0:
        frictionTerm = 0
    else:
        velDifArray.reshape(1,2)
        velDif = perpUnitVect[0] * velDifArray[0] + perpUnitVect[1] * velDifArray[1]
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
    
    if agent.position[0] > obstacle.xMin and agent.position[0] < obstacle.xMax:
        '''
        If the agent is parallel to a wall with constant y then choose the point that
        matches the x coordinate of the agent and the closest wall's y coordinate
        '''
        closestX = agent.position[0]
        
        if abs(agent.position[1]-obstacle.yMin) < abs(agent.position[1]-obstacle.yMax):
            closestY = obstacle.yMin
        else:
            closestY = obstacle.yMax
            
    elif agent.position[1] > obstacle.yMin and agent.position[1] < obstacle.yMax:
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

def socialForceModel():
    '''
    This will run through all of the agents and calculate the total force on
    them from all of the other agents and obstacles.
    '''
    # Subtract one since the index starts from zero
    numOfAgents = Agent.number - 1
    numOfObstacles = Obstacle.number - 1
    # These variables are what the loop runs through while calculating forces
    currentAgent = 0
    otherAgent = 1
    obstacle = 0
    totalForce = 0
    
    # This block of code runs through all of the agents and calculates the 
    # total force on an agent each time.  One thing that I'd like to add
    # is to use the equal and opposite force while going through the agents
    while currentAgent <= numOfAgents:
        goalForce = np.array([[0],[0]])
        totalForce = np.array([[0],[0]])
        totalAgentForce = np.array([[0],[0]])
        totalObstacleForce = np.array([[0],[0]])
        otherAgent = 0
        obstacle = 0
        
        # Since there is only one goal no loop is needed
        goalForce = agentToGoalForce(Agent.agentList[currentAgent])
        
        # This loop finds the agent to agent forces
        while otherAgent <= numOfAgents:
            if otherAgent == currentAgent:
                otherAgent += 1
                
            else:
                agentForce = agentToAgentForce(Agent.agentList[currentAgent],
                                               Agent.agentList[otherAgent])
                totalAgentForce = np.add(totalAgentForce, agentForce)
                
                otherAgent += 1
        
        # This loop finds the agent to obstacle forces
        while obstacle <= numOfObstacles:    
            obstacleForce = agentToObstacleForce(Agent.agentList[currentAgent],
                                                 Obstacle.obstacleList[obstacle])
            totalObstacleForce = np.add(totalObstacleForce, obstacleForce)
            obstacle += 1
            
        # Calculate the sum of the forces and then update the agent's force vector
        totalForce = np.add(np.add(goalForce, totalAgentForce), totalObstacleForce)
#        totalForce = goalForce
        acceleration = totalForce / Agent.agentList[currentAgent].mass

        
        # Now that the agent has their forces we need to calculate their new
        # velocity and corresponding position with simple kinematics
        # v = v0 + a * t
        # x = x0 + v0 * t + (a * t^2) / 2
        # v0 = current velocity, a is found from the forces, and t is time
        
        newXVelocity = Agent.agentList[currentAgent].velocity[0] + acceleration[0] * timeIncrement
        newXPosition = Agent.agentList[currentAgent].position[0] + Agent.agentList[currentAgent].velocity[0] * timeIncrement + acceleration[0] * timeIncrement**2 / 2.0
        
        newYVelocity = Agent.agentList[currentAgent].velocity[1] + acceleration[1] * timeIncrement
        newYPosition = Agent.agentList[currentAgent].position[1] + Agent.agentList[currentAgent].velocity[1] * timeIncrement + acceleration[1] * timeIncrement**2 / 2.0
        
        Agent.agentList[currentAgent].newVelocity[0] = newXVelocity[0]
        Agent.agentList[currentAgent].newVelocity[1] = newYVelocity[0]
        
        Agent.agentList[currentAgent].newPosition[0] = newXPosition[0]
        Agent.agentList[currentAgent].newPosition[1] = newYPosition[0]
        
#        print 'Agent ' + str(currentAgent) + ' is at position: '
#        print Agent.agentList[currentAgent].position
#        print '\n'
        
        
        currentAgent += 1
    
    # Now that all of the new positions and velocities have been calculated,
    # update all of the agents
    currentAgent = 0
    while currentAgent <= numOfAgents:
        Agent.agentList[currentAgent].velocity[0] = Agent.agentList[currentAgent].newVelocity[0]
        Agent.agentList[currentAgent].velocity[1] = Agent.agentList[currentAgent].newVelocity[1]
        
        Agent.agentList[currentAgent].position[0] = Agent.agentList[currentAgent].newPosition[0]
        Agent.agentList[currentAgent].position[1] = Agent.agentList[currentAgent].newPosition[1]
        
        currentAgent += 1
    
    socialForceModelPlot()

    return None

def socialForceModelPlot():
    '''
    This is a function that calculates a plot of the current layout
    '''
     # Subtract one since the index starts from zero
    numOfAgents = Agent.number - 1
    numOfObstacles = Obstacle.number - 1
    agentXPositions = []
    agentYPositions = []
    currentAgent = 0
    currentObstacle = 0
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    # Create dots for each of the agents by getting all of their x and y
    # positions into two lists and then plotting them after the loop
    while currentAgent <= numOfAgents:
        agentXPositions.append(Agent.agentList[currentAgent].position[0])
        agentYPositions.append(Agent.agentList[currentAgent].position[1])
        currentAgent += 1
    
    plt.scatter(agentXPositions, agentYPositions, color = 'r')
    
    while currentObstacle <= numOfObstacles:
        rWidth = abs(Obstacle.obstacleList[currentObstacle].xMin - Obstacle.obstacleList[currentObstacle].xMax)
        rHeight = abs(Obstacle.obstacleList[currentObstacle].yMin - Obstacle.obstacleList[currentObstacle].yMax)

        ax.add_artist(patches.Rectangle(xy=(Obstacle.obstacleList[currentObstacle].xMin, 
                           Obstacle.obstacleList[currentObstacle].yMin),
                width=rWidth, height=rHeight))
        
        currentObstacle += 1
        
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    return None

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

agentOne = Agent([0, 0], [0, 0], [1,1], [10, 15])
agentTwo = Agent([1, 1], [0, 0], [1,1], [15, 10])
agentThree = Agent([2, 2], [0, 0], [1,1], [10, 5])

wallOne = Obstacle(3,0,5,7)
bottomBoundary = Obstacle(-0.25, -0.25, 20, 0)
leftBoundary = Obstacle(-0.25, -0.25, 0, 20)
topBoundary = Obstacle(0, 20, 20, 20.25)
rightBoundary = Obstacle(20, 0, 20.25, 20)


n = 0
while n < 5:
    socialForceModel()
    n += 1
