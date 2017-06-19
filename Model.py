# -*- coding: utf-8 -*-
"""
Haven't really done much
"""

class Agent:
    '''
    The class that defines the agents
    '''
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
    
        
agentOne = Agent(1, [1,1], [2,2])
print agentOne.xVals()
print agentOne.yVals()