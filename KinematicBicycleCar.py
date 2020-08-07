# Kinematic bicycle car model
# Emiko Soroka
# File created: 08/05/2020
# Code moved from nonlinear_mpc_casadi.ipynb Jupyter notebook

import casadi
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl

class KinematicBicycleCar():

	def __init__(self, N = 30, step = 0.01):
		# We construct the model as a set of differential-algebraic equations (DAE)
		self.dae = casadi.DaeBuilder()
		
		# Parameters
		n = 5 # states
		m = 2 # controls
		self.N    = N
		self.step = step
		self.T   = N*step # Time horizon (seconds)
		# https://link-springer-com.stanford.idm.oclc.org/article/10.1007/s13177-020-00226-1#Tab4

		#a_limit = np.array([-5, 1])
		# Steering limit max/min in radians
		#delta_f_limit = np.array([-np.pi/4, np.pi/4])

		# Constants
		self.lr = 2.10 # distance from CG to back wheel in meters
		self.lf = 2.67
		# Source, page 40: https://www.diva-portal.org/smash/get/diva2:860675/FULLTEXT01.pdf

		# States
		z    = self.dae.add_x('z', 5)
		x    = z[0]
		y    = z[1]
		v    = z[2]
		psi  = z[3]
		beta = z[4]

		self.STATE_NAMES = ['x', 'y', 'v', 'psi', 'beta']
		 
		# Controls
		u       = self.dae.add_u('u',2)       # acceleration
		a       = u[0]
		delta_f = u[1] # front steering angle
		self.CONTROL_NAMES = ['a', 'delta_f']

		# Define ODEs
		xdot = v*casadi.cos(psi + beta)
		ydot = v*casadi.sin(psi + beta)
		vdot = a
		psidot = v/self.lr*casadi.sin(beta)
		betadot = v/(self.lf + self.lr)*casadi.tan(delta_f) - v/self.lr*casadi.sin(beta)

		zdot = casadi.vertcat(xdot, ydot, vdot, psidot, betadot)
		self.dae.add_ode('zdot', zdot)


		# Customize Matplotlib:
		mpl.rcParams['font.size'] = 18
		mpl.rcParams['lines.linewidth'] = 3
		mpl.rcParams['axes.grid'] = True


	# Specify initial conditions
	def set_initial(self, ic:[]):
		for i, s in enumerate(self.dae.x):
			self.dae.set_start(s.name(), ic[i])


	# PLOTTING
	def plot_u(self, u_executed, u_planned):
		fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4), sharex=True, gridspec_kw={'wspace': 0.5})

		# Plot the given data
		tgrid = np.linspace(0, self.T, len(u_executed[0]))
		ax1.step(tgrid, u_executed[0],           label="(executed)")
		ax2.step(tgrid, 180/np.pi*u_executed[1], label="(executed)")

		# Plot the last optimal path computed
		tgrid = np.linspace(tgrid[-1], tgrid[-1]+self.T, self.N)
		ax1.step(tgrid, u_planned[0], '--',           label="(planned)")
		ax2.step(tgrid, 180/np.pi*u_planned[1], '--', label="(planned)")

		ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
		ax1.set(ylabel="Acceleration, m/s^2", xlabel="Time (s)")
		ax2.set(ylabel="Steering angle, deg.", xlabel="Time (s)")
		fig1.suptitle("Control signals")

		return fig1, ax1, ax2

	def plot_x(self, x_executed, x_planned):
		fig2, ax = plt.subplots(1,1, figsize=(12, 4))

		# Plot the last optimal path computed
		tgrid = np.linspace(0, self.T, self.N)
		vx = np.multiply(x_planned[2],np.cos(x_planned[3]))
		vy = np.multiply(x_planned[2],np.sin(x_planned[3]))
		q = ax.quiver(x_planned[0], x_planned[1], vx, vy, color='orange', linewidth=0.5)
		ax.quiverkey(q, X=0.25, Y=0.2, U=5,
					 label='Planned', labelpos='E')

		# Plot the x given
		tgrid = np.linspace(0, self.T, len(x_executed[0]))
		vx = np.multiply(x_executed[2],np.cos(x_executed[3]))
		vy = np.multiply(x_executed[2],np.sin(x_executed[3]))
		q = ax.quiver(x_executed[0], x_executed[1], vx, vy, color='blue', linewidth=0.5)
		ax.quiverkey(q, X=0.25, Y=0.1, U=5,
					 label='Velocity', labelpos='E')

		ax.scatter(x_executed[0], x_executed[1], color='navy')
		ax.scatter(x_planned[0], x_planned[1], color='red')

		ax.set(xlabel="x (m)",ylabel="y (m)")
		fig2.suptitle("Trajectory")

		return fig2, ax


