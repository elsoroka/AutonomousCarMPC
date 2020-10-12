# Kinematic bicycle car model
# Emiko Soroka
# File created: 08/05/2020
# Code moved from nonlinear_mpc_casadi.ipynb Jupyter notebook

import casadi
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
from AbstractBaseCar import AbstractBaseCar

class KinematicBicycleCar(AbstractBaseCar):

	def __init__(self, N = 30, step = 0.01):
		# We construct the model as a set of differential-algebraic equations (DAE)
		self.dae = casadi.DaeBuilder()
		
		# Parameters
		self.n = 4
		self.m = 2 # controls
		self.N    = N
		self.step = step
		self.T   = N*step # Time horizon (seconds)

		self.fixed_points = dict() # None yet

		# Constants
		self.lr = 2.10 # distance from CG to back wheel in meters
		self.lf = 2.67
		# Source, page 40: https://www.diva-portal.org/smash/get/diva2:860675/FULLTEXT01.pdf

		# States
		z    = self.dae.add_x('z', self.n)
		x    = z[0]
		y    = z[1]
		v    = z[2]
		psi  = z[3]

		self.STATE_NAMES = ['x', 'y', 'v', 'psi',]
		 
		# Controls
		u       = self.dae.add_u('u',self.m)       # acceleration
		a       = u[0]
		delta_f = u[1] # front steering angle
		
		# This is a weird "state".
		beta = casadi.arctan(self.lr/(self.lr + self.lf)*casadi.tan(delta_f))

		self.CONTROL_NAMES = ['a', 'delta_f']

		# Define ODEs
		xdot = v*casadi.cos(psi + beta)
		ydot = v*casadi.sin(psi + beta)
		vdot = a
		psidot  = v/self.lr*casadi.sin(beta)

		zdot = casadi.vertcat(xdot, ydot, vdot, psidot)
		self.dae.add_ode('zdot', zdot)


		# Customize Matplotlib:
		mpl.rcParams['font.size'] = 18
		mpl.rcParams['lines.linewidth'] = 3
		mpl.rcParams['axes.grid'] = True

		self.state_estimate = None
		self.control_estimate = np.zeros((self.m, self.N))



	def getDae(self)->casadi.DaeBuilder:
		return self.dae

	# Specify initial conditions
	def set_initial(self, DESIRED_SPEED, roadrunner):
		# For SOME REASON this doesn't work.
		#self.dae.set_start(self.dae.x[0], ic)
		state = roadrunner.save_state()
		def desired_speed(k:int): # TODO: HACK. FIX
			return DESIRED_SPEED
		self.desired_speed = desired_speed

		if self.state_estimate is None:
			self.state_estimate = np.empty((self.n,self.N+1))
			for i in range(self.N):
				(xy, psi, _) = roadrunner.evaluate(full_data=True)
				# x,y
				self.state_estimate[0:2,i] = xy
				self.state_estimate[2,i]   = self.desired_speed(i)
				self.state_estimate[3]     = psi
				roadrunner.advance(self.desired_speed(i)*self.step)

		roadrunner.reset(**state)


	def set_fixed_point(self, k:int, fixed_upper:np.array, fixed_lower:np.array)->None:
		self.fixed_points[k] = [fixed_upper, fixed_lower]
		if np.isfinite(fixed_upper[2]) and np.isfinite(fixed_lower[2]):
			self.state_estimate[2,k] = 0.5*(fixed_upper[2] + fixed_lower[2]) # average.

	def clear_fixed_point(self, k:int or [int])->None:
		if typeof(k) == int:
			del self.fixed_points[k]
			return
		# Delete multiple keys.
		for key in k:
			del self.fixed_points[k]
		# Will crash if k not iterable.

	def set_state_estimate(self, state_estimate:np.array):
		# The state estimate is the previous optimal state
		# so we take the state at time k+1 for our new time k
		self.state_estimate[:,:-1] = state_estimate[:,1:]
		# The last one, we don't know. So we guess it's
		# the second-to-last one.
		self.state_estimate[:,-1] = self.state_estimate[:,-2]

	def set_control_estimate(self, control_estimate:np.array):
		# The control estimate is the previous optimal control
		# so we take the control at time k+1 for our new time k
		self.control_estimate[:,:-1] = control_estimate[:,1:]
		# The last one, we don't know. So we guess it's
		# the second-to-last one.
		self.control_estimate[:,-1] = np.zeros((self.m,)) #self.control_estimate[:,-2]
	
	
	# CONSTRAINT HANDLERS
	def upperbounds_x(self, k:int)->np.array:
		# This function returns the upper bound for the entire state vector.
		# at given index k=0,...,N with velocity estimate v_estimate
		# (this is important because we may want to estimate our position)
		# These functions should take into account any fixed points.
		if k in self.fixed_points.keys():
			return self.fixed_points[k][0] # Recall upper is first, then lower
		else:
			return np.array([np.inf,
                  			np.inf,
                  			50.0,
                  			np.pi,
							])

	def lowerbounds_x(self, k:int)->np.array:
		if k in self.fixed_points.keys():
			return self.fixed_points[k][1] # Recall upper is first, then lower
		else:
			return np.array([-np.inf,
                  			 -np.inf,
                 			  0.0,
                 			 -np.pi,
                 			 ])


	def upperbounds_u(self, k:int)->np.array:
		return np.array([2.5, np.pi/4]) # <= 0.84 preferred

	def lowerbounds_u(self, k:int)->np.array:
		return np.array([-5, -np.pi/4]) # >= -1.70 preferred


	# PLOTTING
	def plot_u(self, u_executed:np.array, u_planned:np.array):
		fig1, (ax1, ax2) = plt.subplots(1, 2,
			figsize=(12,4), sharex=True,
			gridspec_kw={'wspace': 0.5})

		# Plot the given data
		tgrid = np.linspace(0, self.T, len(u_executed[0]))
		ax1.step(tgrid, u_executed[0],           label="(executed)")
		ax2.step(tgrid, 180/np.pi*u_executed[1], label="(executed)")

		# Plot the last optimal path computed
		tgrid = np.linspace(tgrid[-1], tgrid[-1]+self.T, self.N)
		ax1.step(tgrid, u_planned[0], '--',           label="(planned)")
		ax2.step(tgrid, 180/np.pi*u_planned[1], '--', label="(planned)")

		# Fix the legend and labels
		ax2.legend(bbox_to_anchor=(1.05, 1),
				   loc='upper left',
				   borderaxespad=0.0)
		ax1.set(ylabel="Acceleration, m/s^2",  xlabel="Time (s)")
		ax2.set(ylabel="Steering angle, deg.", xlabel="Time (s)")
		fig1.suptitle("Control signals")

		return fig1, ax1, ax2


	def plot_x(self, x_executed:np.array, x_planned:np.array):
		fig2, ax = plt.subplots(1,1,
			figsize=(14, 6))

		# Plot the last optimal path computed
		tgrid = np.linspace(0, self.T, self.N)
		vx = np.multiply(x_planned[2],np.cos(x_planned[3]))
		vy = np.multiply(x_planned[2],np.sin(x_planned[3]))
		q = ax.quiver(x_planned[0], x_planned[1], vx, vy,
					  color='orange', linewidth=0.5,)
		ax.quiverkey(q, X=0.25, Y=0.2, U=5,
					 label='Planned', labelpos='E')

		# Plot the x given
		tgrid = np.linspace(0, self.T, len(x_executed[0]))
		vx = np.multiply(x_executed[2],np.cos(x_executed[3]))
		vy = np.multiply(x_executed[2],np.sin(x_executed[3]))
		q = ax.quiver(x_executed[0], x_executed[1], vx, vy,
					  color='blue', linewidth=0.5,)
		ax.quiverkey(q, X=0.25, Y=0.1, U=5,
					 label='Velocity', labelpos='E')

		# Plot position dots over the arrows. It looks better.
		ax.scatter(x_executed[0], x_executed[1], color='navy')
		ax.scatter(x_planned[0],  x_planned[1],  color='red')

		ax.set(xlabel="x (m)",ylabel="y (m)")
		fig2.suptitle("Trajectory")

		return fig2, ax

	def plot_with_time(self, x_executed:np.array, x_planned:np.array, boundary_up:np.array, boundary_low:np.array):
		fig1, (ax1, ax2) = plt.subplots(2,1, figsize=(10,4), sharex=True)

		# Plot the executed steps
		t = np.linspace(0, 0.05*(np.size(x_executed)//self.n-1), np.size(x_executed)//self.n)
		ax1.plot(t, x_executed[0,:], color='blue')
		ax2.plot(t, x_executed[1,:], color='blue')

		# Plot the planned steps
		t = t[-1] + np.linspace(0, 0.05*(np.size(x_planned)//self.n-1), np.size(x_planned)//self.n)
		ax1.plot(t, x_planned[0,:], color='orange')
		ax2.plot(t, x_planned[1,:], color='orange')

		# Plot the bounds around the executed steps
		t = np.linspace(0, 0.05*(np.size(boundary_up)//2-1), np.size(boundary_up)//2)
		ax1.plot(t, boundary_up[:,0], color='gray')
		ax1.plot(t, boundary_low[:,0], color='gray')
		ax2.plot(t, boundary_up[:,1], color='gray')
		ax2.plot(t, boundary_low[:,1], color='gray')

		ax1.set(ylabel="x (m)")
		ax2.set(ylabel="y (m)", xlabel="Time (s)")
		return fig1, ax1, ax2

