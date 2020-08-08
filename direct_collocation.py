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
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class MpcProblem:

    def __init__(self, model, cost):        # casadi symbolic objective

        self.model = model
        self.sys  = model.getDae()
        self.cost = cost
        params = {'x':self.sys.x[0], 'p':self.sys.u[0], 'ode':self.sys.ode[0]}
        self.sim = ca.integrator('F', 'idas', params, {'t0':0, 'tf':model.step})

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
            lbw.append(np.reshape(self.model.lowerbounds_u(k), (self.model.m,)))
            ubw.append(np.reshape(self.model.upperbounds_u(k), (self.model.m,)))
            w0.append(np.zeros((self.model.m,)))
            u_plot.append(Uk)

            # State at collocation points
            Xc = []
            for j in range(self._d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), self.model.n)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(np.reshape(self.model.lowerbounds_x(k), (self.model.n,)))
                ubw.append(np.reshape(self.model.upperbounds_x(k), (self.model.n,)))
                w0.append(np.zeros((self.model.n,)))

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
            lbw.append(np.reshape(self.model.lowerbounds_x(k), (self.model.n,)))
            ubw.append(np.reshape(self.model.upperbounds_x(k), (self.model.n,)))
            w0.append(np.zeros((self.model.n, )))
            x_plot.append(Xk)

            # Add equality constraint
            g.append(Xk_end-Xk)
            lbg.append(np.zeros((self.model.n,)))
            ubg.append(np.zeros((self.model.n,)))

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

        # Feed the velocity at N = 30 points back to the model
        self.model.set_v_estimate(self.x_opt[2])

        return self.x_opt, self.u_opt

    def simulate(self, x:np.array, u:np.array):
      r = self.sim(x0=x, p=u)
      xf = r['xf']
      return np.reshape(xf, (self.model.n,))

