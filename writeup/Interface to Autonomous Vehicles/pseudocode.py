import numpy as np

class TrajectoryPlanner():
	def __init__(self, **params):
		# params: hardware-specific values
		# (such as vehicle geometry, engine power, inertia...)
		# MPC parameters (timestep size, number of lookahead steps)
		# etc.
		self.params = params


	def __run__(self, initial_state:np.array,
					  driveable_corridor:callable,
					  desired_speed:callable,
					  constraint_generator:callable):
		self.z0 = initial_state
		self.initialize_first_mpc_problem(driveable_corridor, desired_speed)

		while True:
			problem = self.build_mpc_problem(driveable_corridor, desired_speed, constraint_generator)
			z, u = self.solve_mpc_problem(problem)
			# Move forward one timestep
			self.z0 = self.apply_control(u[0])

	# TO DO TODAY: Code Rewrite
	# Bring code into compliance with paper
	# Implement front-following case
	# Write pseudocode