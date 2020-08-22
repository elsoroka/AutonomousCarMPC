#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#


# File MODIFIED from example direct_collocation.py provided with CasADi
# to manage an MPC problem using a CasADI dynamics model and symbolic cost
# using direct collocation to handle the dynamics

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class MpcProblem:

    def __init__(self, model, cost, # casadi symbolic objective
                 lowerbounds_x, upperbounds_x, # callables
                 lowerbounds_u, upperbounds_u, # callables
                 road_center, # callable
                 ): 

        self.model = model
        self.sys  = model.getDae()
        self.cost = cost
        self.lowerbounds_x = lowerbounds_x
        self.upperbounds_x = upperbounds_x
        self.lowerbounds_u = lowerbounds_u
        self.upperbounds_u = upperbounds_u
        self.road_center   = road_center
        
        params = {'x':self.sys.x[0], 'p':self.sys.u[0], 'ode':self.sys.ode[0]}
        self.sim = ca.integrator('F', 'cvodes', params, {'t0':0, 'tf':model.step})

        # Set up collocation (from direct_collocation example)

        # Degree of interpolating polynomial
        self._d = 3

        # Get collocation points
        tau_root = np.append(0, ca.collocation_points(self._d, 'legendre'))

        # Coefficients of the collocation equation
        self._C = np.zeros((self._d+1,self._d+1))

        # Coefficients of the continuity equation
        self._D = np.zeros(self._d+1)

        # Coefficients of the quadrature function
        self._B = np.zeros(self._d+1)

        # Construct polynomial basis
        for j in range(self._d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(self._d+1):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            self._D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(self._d+1):
                self._C[j,r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            self._B[j] = pint(1.0)


    # Update the cost function between iterations
    def set_cost(self, cost):
        self.cost = cost


    def run(self, ic:np.array):
        self.ic = np.reshape(ic, (self.model.n,))
        # Start with an empty NLP
        w   = [] # state
        w0  = [] # initial state
        lbw = [] # lower bound on state
        ubw = [] # upper bound on state
        J = 0
        g   = [] # constraint
        lbg = [] # lower bound on constraint
        ubg = [] # upper bound on constraint

        # For plotting x and u given w
        x_plot = []
        u_plot = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('z0', self.model.n)
        w.append(Xk)
        lbw.append(self.ic)
        ubw.append(self.ic)
        w0.append(self.ic)
        x_plot.append(Xk)

        self.attractive_cost = 0.0
        self.jerk_cost = 0.0
        self.steering_change_cost = 0.0
        self.Uk_prev = None

        # Create a matrix function
        # HACK: breaks if multiple state variables instead of one vector state
        # same for multiple control variables instead of one vector control!
        f = ca.Function('f', [self.sys.x[0], self.sys.u[0]],
                        [self.sys.ode[0], self.cost],
                        ['x', 'u'],['xdot', 'L'])


        # Formulate the NLP
        for k in range(self.model.N):
            # New NLP variable for the control
            Uk = ca.MX.sym("U_"+str(k), self.model.m)
            w.append(Uk)
            lbw.append(np.reshape(self.lowerbounds_u(self.model, k), (self.model.m,)))
            ubw.append(np.reshape(self.upperbounds_u(self.model, k), (self.model.m,)))
           # w0.append(np.zeros((self.model.m,)))
            w0.append(self.model.control_estimate[:,k])
            u_plot.append(Uk)

            # Add to control change costs
            if None != self.Uk_prev:
                self.jerk_cost += (Uk[0]-self.Uk_prev[0])**2
                self.steering_change_cost += (Uk[1]-self.Uk_prev[1])**2

            self.Uk_prev = Uk

            # State at collocation points
            Xc = []
            for j in range(self._d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), self.model.n)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(np.reshape(self.lowerbounds_x(self.model, k), (self.model.n,)))
                ubw.append(np.reshape(self.upperbounds_x(self.model, k), (self.model.n,)))
                #w0.append(np.zeros((self.model.n,)))
                w0.append(self.model.state_estimate[:,k])

            # Loop over collocation points
            Xk_end = self._D[0]*Xk
            for j in range(1,self._d+1):
               # Expression for the state derivative at the collocation point
               xp = self._C[0,j]*Xk
               for r in range(self._d): xp = xp + self._C[r+1,j]*Xc[r]

               # Append collocation equations
               fj, qj = f(Xc[j-1],Uk)
               g.append(self.model.step*fj - xp)
               lbg.append(np.zeros((self.model.n,)))
               ubg.append(np.zeros((self.model.n,)))

               # Add contribution to the end state
               Xk_end = Xk_end + self._D[j]*Xc[j-1];

               # Add contribution to quadrature function
               J = J + self._B[j]*qj*self.model.step

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.model.n)
            w.append(Xk)
            lbw.append(np.reshape(self.lowerbounds_x(self.model, k+1), (self.model.n,)))
            ubw.append(np.reshape(self.upperbounds_x(self.model, k+1), (self.model.n,)))
            #w0.append(np.zeros((self.model.n, )))
            w0.append(self.model.state_estimate[:,k+1])
            x_plot.append(Xk)

            # Add equality constraint
            g.append(Xk_end-Xk)
            lbg.append(np.zeros((self.model.n,)))
            ubg.append(np.zeros((self.model.n,)))

            # Weakly attract state to middle of road
            xy_k = self.road_center(self.model, k+1)

            self.attractive_cost += ((Xk[0]-xy_k[0])**2 + \
                                     (Xk[1]-xy_k[1])**2 + \
                                     (Xk[3]-xy_k[2])**2)
            print("Attracting ", Xk[0], k+1, "to ", xy_k)

        
        # This attracts the car to the middle of the road
        cost = 10.0*self.attractive_cost + \
               1.0*self.jerk_cost + \
               10.0*180/np.pi*self.steering_change_cost
               # Several papers make this really big


        f = ca.Function('f', [self.sys.x[0], self.sys.u[0]],
                    [self.sys.ode[0], cost],
                    ['x', 'u'],['xdot', 'L'])

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)
        
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create an NLP solver
        prob = {'f': J, 'x': w, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, {'verbose':False});

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        x_opt, u_opt = trajectories(sol['x'])
        
        self.x_opt = x_opt.full() # to numpy array
        self.u_opt = u_opt.full() # to numpy array

        # Feed the previous stateback to the model
        self.model.set_state_estimate(self.x_opt)
        self.model.set_control_estimate(self.u_opt)
        self.model.c += 1

        return self.x_opt, self.u_opt

    def simulate(self, x:np.array, u:np.array):
      r = self.sim(x0=x, p=u)
      xf = r['xf']
      return np.reshape(xf, (self.model.n,))

