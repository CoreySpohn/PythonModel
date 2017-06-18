# PythonModel
A program that simulates crowd dynamics, particularly in a panic situation, using a multi-agent system approach.

# SOFTWARE USED

Python 2.7
	The Anaconda package is highly recomended as it comes with the Spyder IDE and many packages meant to handle large amounts of data.
	
	Download here: https://www.continuum.io/downloads
	
	Python is designed to be very readable, which is the main reason it was chosen.  The previous model was not documented well, so hopefully writing it in python will help future teams.
	
# NAMING CONVENTIONS

## Variables - mixedCase
	I think this is easily readable and looks better than underscores.
	Example:
		agentOnePosition = [19,30]
		
## Functions - mixedCase
	Same as above
	Example:
		def setAgentPosition(agentNumber):
			agentNumberPosition = [19,30]
			
## Classes - CapWords
	Makes it easy to tell that the method (a function that is defined in a class) being called comes from a class
	
For more https://www.python.org/dev/peps/pep-0008/

# Methodology
## Create agents
	I think this should be done with an agent class
	Assign it x/y coordinates
		Preferably it will be in a 2 number list
			Example
				agentOnePosition = [5.44,8.32]
			I think this would be more compact
		Could also use two variables 
			Example
				agentOneX=5.44
       			agentOneY=8.32
			This 
		
## Process the forces
	Forces
		Agent-agent
		Agent-environment
		Agent-robot
		Agent-goal
			This is what I'm calling the self-driven force
			Hopefully it will be easy to modify the model into a working sharks and minnows simulator
			
	Forces will be doubles treated as vectors
		Example:
			agent1 acting on agent3 force = [-2.32,5.14]
		
	Agent-agent forces will be equal and opposite so assign the force to the other agent
		Example:
			agent3 acting on agent1 force = [2.32,-5.14]
		
	Sum all of the forces for the agent
		The final force vector will define their movement direction
		QUESTION: Do we just correlate it to their acceleration?

## Update the agent positions
	Move the agents in the based on their final force vector
	Check to make sure the agent is within the region
		If not then move them as far as possible
	