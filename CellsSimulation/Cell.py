import enum
from sympy import *
from sympy.geometry import *
import matplotlib.pylab as plt

class CreateType(enum.Enum):
	BOUNDARIES = 0
	#SOURCE
	
class State(enum.Enum):
	STATIC = 0
	GROWING = 1
	SPLITTING = 2
	FINISHED_SPLITING = 3
	DEAD = 4
	REMOVED = 5

class Cell:

	toRemoved = False
	general_status = {}
	
	def __init__(self, 
				cell_create_type = 0, 
				points = [],
				growthRate = 1):

		self.growth_rate = growthRate
		if (cell_create_type == CreateType.BOUNDARIES):
			self._boundries = points
	
	def getBoundries(self):
		#print(self.boundries)
		return self._boundries

	def remove(self):
		self.toRemoved = True

	def _grow(self):
		pass

	def updateCell(self):
		pass 
