import numpy as np
from KinematicBicycleCar import KinematicBicycleCar
from direct_collocation import MpcProblem

class TrajectoryPlanner():
	def __init__(self, **params):
		# params: hardware-specific values
		# (such as vehicle geometry, engine power, inertia...)
		# MPC parameters (timestep size, number of lookahead steps)
		# etc.
		self.N = params["N"] # number of steps
		self.n = params["n"] # number of states
		self.m = params["m"] # number of controls
		self.step = params["step"]
		self.extraparams = params


		# This is the model:
		# https://link-springer-com.stanford.idm.oclc.org/article/10.1007/s13177-020-00226-1/figures/3
		self.mpcprob = MpcProblem(params)


	def run(self, initial_state:np.array,
					  driveable_corridor:callable,
					  desired_speed:callable,
					  constraint_generator:callable,
					  weights:dict,
					  n_runs=100):

		self.setup_to_save_data(n_runs)
		self.z0 = initial_state
		estimated_path, _, _ = self.generate_path_estimate(driveable_corridor, desired_speed, constraint_generator)
		self.initialize_first_mpc_problem(estimated_path)

		for r in range(n_runs):
			# Construct the MPC problem
			estimated_path, left_widths, right_widths = self.generate_path_estimate(driveable_corridor, desired_speed, constraint_generator)
			problem = self.build_mpc_problem(estimated_path, left_widths, right_widths, weights)

			# save data
			_,p = self.mpcprob.bound_x(estimated_path[0,r], estimated_path[1,r], estimated_path[3,r], left_widths[r], right_widths[r])
			self.polygon_boundaries[:,:,r] = p

			# Compute the trajectory z and control u
			z, u, info  = self.solve_mpc_problem(problem)
			u0 = np.array([u[0][0], u[1][0]])
			# Move forward one timestep
			self.z0 = self.apply_control(u0)
			# Initialize the next problem with the result of the previous
			self.initialize_nth_mpc_problem(z, u)

	# TO DO TODAY: Code Rewrite
	# Bring code into compliance with paper
	# Implement front-following case
	# Write pseudocode

	def generate_path_estimate(self, driveable_corridor, desired_speed, constraint_generator):
		# Use the callables to step along the road
		z_estimate = np.zeros((self.n, self.N+1))
		z_estimate[:,0] = self.z0
		dl, dr = np.zeros(self.N+1), np.zeros(self.N+1)
		for k in range(1, self.N+1):
			x_k, y_k, psi_k, dl_k, dr_k = driveable_corridor(self.mpcprob.model.state_estimate[0,k-1], \
				                                             self.mpcprob.model.state_estimate[1,k-1], \
				                                             self.mpcprob.model.step*self.mpcprob.model.state_estimate[2,k-1])
			v_k = desired_speed(x_k, y_k, k)
			z_estimate[:,k] = np.array([x_k, y_k, v_k, psi_k])
			dl[k] = dl_k
			dr[k] = dr_k
		dl[0] = dl[1]; dr[0] = dr[1]
		return z_estimate, dl, dr

	def generate_first_estimate(self, ic, driveable_corridor, desired_speed, constraint_generator):

		state_estimate = np.zeros((self.n,self.N+1))
		control_estimate = np.zeros((self.m,self.N))
		self.z0 = ic
		s = 0
		for i in range(self.N+1):
			(x, y, psi, _, _) = driveable_corridor(self.z0[0], self.z0[1], s)
			# x,y
			v_des = desired_speed(i, x, y)
			state_estimate[0, i] = x
			state_estimate[1,i] = y
			state_estimate[2,i]   = v_des
			state_estimate[3,i]   = psi
			s += v_des*self.step
		
			# Very bad rough estimate of acceleration
			if i < self.N:
				x, y, psi, _, _ = driveable_corridor(self.z0[0], self.z0[1], s)
				control_estimate[0,i] = (desired_speed(x, y, i+1) - v_des)/(self.step)

		return state_estimate, control_estimate

	def initialize_first_mpc_problem(self, estimated_path):
		# save the data
		self.z0 = estimated_path[:,0] # initial condition
		self.centers[:,0] = self.z0 # stupid indexing difference, sorry
		self.x_true[:,0] = self.z0 # initial conditions
		self.x_plan[:,0] = self.z0

		# Very bad rough estimate of acceleration
		estimated_control = np.zeros((self.m, self.N))
		for i in range(self.N-1):
			estimated_control[0,i] = (estimated_path[2,i+1]-estimated_path[2,i])/self.step
			estimated_control[1,i] = (estimated_path[3,i+1]-estimated_path[3,i])/self.step

		#self.mpcprob.model.set_initial(estimated_path, estimated_control)

	def initialize_nth_mpc_problem(self, z, u):
		pass
		#self.mpcprob.model.set_state_estimate(z)
		#self.mpcprob.model.set_control_estimate(u)

	def build_mpc_problem(self, estimated_path, left_widths, right_widths, weights):
		problem = self.mpcprob.build_problem(self.z0, estimated_path, left_widths, right_widths, weights)
		return problem

	def solve_mpc_problem(self, problem):
		xk_opt, uk_opt, sol = self.mpcprob.solve(problem)
		return xk_opt, uk_opt, sol

	def apply_control(self, u):
		return self.mpcprob.simulate(self.z0, u)

	def setup_to_save_data(self, n_runs=100):
		# Set up to save n_runs of data
		self
		self.x_plan = np.empty((self.mpcprob.model.n,n_runs+2)) # store the steps that get executed (n_runs)
		self.u_plan = np.empty((self.mpcprob.model.m,n_runs+1))   # store the control inputs that get executed
		self.x_true = np.empty((self.mpcprob.model.n,n_runs+2)) # store the state as simulated for each control input by an integrator
		self.centers = np.zeros((4,n_runs+2)) # store the state as simulated for each control input by an integrator
		# store the polygon boundary for each step, so we can plot them later
		self.polygon_boundaries = np.zeros((n_runs+self.N,4,2))
