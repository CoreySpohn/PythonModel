# PythonModel
A program that simulates crowd dynamics, particularly in a panic situation, using a multi-agent system approach.

# Software Used

## Python 2.7

Python is designed to be very readable, which is the main reason it was chosen.  The previous model was not documented well, so hopefully writing it in Python will help future teams.

We're using Python 2 because there are still some packages that haven't been ported to Python 3. We're not sure which packages we'll be using and don't want to have to rewrite our model if we need a specific package that hasn't been ported yet.

## Anaconda

Anaconda is a Python package designed specifically for data analysis. It comes with Spyder, a MATLAB like Python IDE, as well as the most common Python packages for data science.

Download Anaconda here: https://www.continuum.io/downloads
	
# Naming Conventions

## Variables - mixedCase
I think this is easily readable and looks better than underscores.

Example:

	agentOnePosition = [19,30]
		
## Functions - mixedCase
Typically the function and variables follow the same conventions

Example:

	def setAgentPosition(agentNumber):
	
		agentNumberPosition = [19,30]
			
## Classes - CapWords
Makes it easy to tell that the method (a function that is defined in a class) being called comes from a class
	
For more https://www.Python.org/dev/peps/pep-0008/

# Methodology
## Create agents

I think this should be done with an agent class

Assign it x/y coordinates

	Preferably it will be in a 2 number list
	
		Example
		
			agentOnePosition = [5.44,8.32]
			
		I think this is more compact and more functional
		
	Could also use two variables 
	
		Example
		
			agentOneX=5.44
			
			agentOneY=8.32
		
## Process the forces

I'd like to use the Headed Social Force Model (HSFM) because it combines the Social Force Model with the Human Locomotion Model, ideally I'll be able to program all three and we can test them independently

Forces

	Agent-agent
	
	Agent-environment
	
	Agent-robot
	
	Agent-goal
	
		This is what I'm calling the self-driven force
		
Forces will be doubles treated as vectors

	Example:
	
		agent1 acting on agent3 force = [-2.32,5.14]
	
Agent-agent forces will be equal and opposite so assign the force to the other agent

	Example:
	
		agent3 acting on agent1 force = [2.32,-5.14]
	
Sum all of the forces for the agent

	The final force vector will define their movement direction
	
	QUESTION: Do we just correlate it to their acceleration

## Update the agent positions
Move the agents in the based on their final force vector

Check to make sure the agent is within the region

	If not then move them as far as possible
	