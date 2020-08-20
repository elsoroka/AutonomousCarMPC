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
		self.n = 5 # states
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
		beta = z[4]

		self.STATE_NAMES = ['x', 'y', 'v', 'psi', 'beta']
		 
		# Controls
		u       = self.dae.add_u('u',self.m)       # acceleration
		a       = u[0]
		delta_f = u[1] # front steering angle

		self.CONTROL_NAMES = ['a', 'delta_f']

		# Define ODEs
		xdot = v*casadi.cos(psi + beta)
		ydot = v*casadi.sin(psi + beta)
		vdot = a
		psidot  = v/self.lr*casadi.sin(beta)
		betadot = v/(self.lf + self.lr)*casadi.tan(delta_f) - \
		          v/self.lr*casadi.sin(beta)

		zdot = casadi.vertcat(xdot, ydot, vdot, psidot, betadot)
		self.dae.add_ode('zdot', zdot)


		# Customize Matplotlib:
		mpl.rcParams['font.size'] = 18
		mpl.rcParams['lines.linewidth'] = 3
		mpl.rcParams['axes.grid'] = True

		self.state_estimate = None
		self.c = 0


	def getDae(self)->casadi.DaeBuilder:
		return self.dae

	# Specify initial conditions
	def set_initial(self, ic:[]):
		for i, s in enumerate(self.dae.x):
			self.dae.set_start(s.name(), ic[i])
		if None == self.state_estimate:
			self.state_estimate = np.empty((5,self.N+1))
			for i in range(5):
				self.state_estimate[i,:] = ic[i]*np.ones(self.N+1)


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
		self.state_estimate = state_estimate
	
	
	# CONSTRAINT HANDLERS
	def upperbounds_x(self, k:int)->np.array:
		# This function returns the upper bound for the entire state vector.
		# at given index k=0,...,N with velocity estimate v_estimate
		# (this is important because we may want to estimate our position)
		# These functions should take into account any fixed points.
		if k in self.fixed_points.keys():
			return self.fixed_points[k][0] # Recall upper is first, then lower
		else:
			return np.array([self.state_estimate[0,0] + self.step*np.sum(self.state_estimate[2,:k])+1,#20.0,
                  			self.state_estimate[1,0] + np.sin((k+self.c)*np.pi/60)+1,#1.0,
                  			20.0,
                  			np.pi/4,
                  			np.pi/4])

	def lowerbounds_x(self, k:int)->np.array:
		if k in self.fixed_points.keys():
			return self.fixed_points[k][1] # Recall upper is first, then lower
		else:
			return np.array([self.state_estimate[0,0] + self.step*np.sum(self.state_estimate[2,:k])-1,#-1.0,
                  			 self.state_estimate[1,0] + np.sin((k+self.c)*np.pi/60)-1,#-1.0,
                 			  0.0,
                 			 -np.pi/4,
                 			 -np.pi/4])


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
			figsize=(12, 4))

		# Plot the last optimal path computed
		tgrid = np.linspace(0, self.T, self.N)
		vx = np.multiply(x_planned[2],np.cos(x_planned[3]))
		vy = np.multiply(x_planned[2],np.sin(x_planned[3]))
		q = ax.quiver(x_planned[0], x_planned[1], vx, vy,
					  color='orange', linewidth=0.5)
		ax.quiverkey(q, X=0.25, Y=0.2, U=5,
					 label='Planned', labelpos='E')

		# Plot the x given
		tgrid = np.linspace(0, self.T, len(x_executed[0]))
		vx = np.multiply(x_executed[2],np.cos(x_executed[3]))
		vy = np.multiply(x_executed[2],np.sin(x_executed[3]))
		q = ax.quiver(x_executed[0], x_executed[1], vx, vy,
					  color='blue', linewidth=0.5)
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

		# Plot the bounds around the executed steps
		t = np.linspace(0, 0.05*(np.size(boundary_up)//2-1), np.size(boundary_up)//2)
		ax1.plot(t, boundary_up[:,0], color='gray')
		ax1.plot(t, boundary_low[:,0], color='gray')
		ax2.plot(t, boundary_up[:,1], color='gray')
		ax2.plot(t, boundary_low[:,1], color='gray')

		# Plot the executed steps
		t = np.linspace(0, 0.05*(np.size(x_executed)//5-1), np.size(x_executed)//5)
		ax1.plot(t, x_executed[0,:], color='blue')
		ax2.plot(t, x_executed[1,:], color='blue')

		# Plot the planned steps
		t = t[-1] + np.linspace(0, 0.05*(np.size(x_planned)//5-1), np.size(x_planned)//5)
		ax1.plot(t, x_planned[0,:], color='orange')
		ax2.plot(t, x_planned[1,:], color='orange')

		ax1.set(ylabel="x (m)")
		ax2.set(ylabel="y (m)", xlabel="Time (s)")
		return fig1, ax1, ax2

