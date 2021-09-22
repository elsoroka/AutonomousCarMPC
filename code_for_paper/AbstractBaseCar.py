# DO NOT USE THIS CLASS
# This is an interface definition for child classes
# implementing kinematic or dynamic car models
# that can be used with the MpcProblem class.
# File created: Aug. 7th, 2020
# Emiko Soroka

import numpy as np

class AbstractBaseCar():
	def __init__(self, N = 30, step = 0.01):
		# Here, you would build the system model
		# for example, using casadi.DaeBuilder
		self.N    = N
		self.step = step
		self.T   = N*step # Time horizon (seconds)
		# Define self.n, the number of states
		# and self.m, the number of control signals

	def set_initial(self, ic:[])->None:
		pass
		# Here you would set the initial conditions.
		# ic must be of length self.n

	def set_fixed_point(self, k:int, fixed_upper:np.array, fixed_lower:np.array)->None:
		pass
		# Here you would set a "fixed point", which is
		# a specific constraint at index k.
		# k has range 1,...,self.N-1
		# To set a constraint at k=0 use set_initial.

	def clear_fixed_point(self, k:int or [int])->None:
		pass
		# Delete one or more fixed points
	
	# CONSTRAINT HANDLERS
	def upperbounds_x(self, k:int, v_estimate:float)->np.array:
		pass
		# This function returns the upper bound for the entire state vector.
		# at given index k=0,...,N with velocity estiamte v_estimate
		# (this is important because we may want to estimate our position)
		# These functions should take into account any fixed points.
	
	def lowerbounds_x(self, k:int, v_estimate:float)->np.array:
		pass
		# Like upperbounds_x but it returns the lower bound.

	def upperbounds_u(self, k:int, v_estimate:float)->np.array:
		pass
		# same

	def lowerbounds_u(self, k:int, v_estimate:float)->np.array:
		pass
		#same


	# PLOTTING
	def plot_u(self, u_executed:np.array, u_planned:np.array):
		pass
		# Here you would produce a good visualization
		# of the control signal(s).
		# u_executed is the signals that have already happened.
		# u_planned is the signals we planned in the latest MPC run.
		# Return the figure object.

	def plot_x(self, x_executed:np.array, x_planned:np.array):
		pass
		# same as plot_u for x.
		# Return the figure object.
