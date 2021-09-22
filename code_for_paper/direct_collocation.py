# Original notes from direct_collocation.py example:
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


# File MODIFIED from example direct_collocation.py
# to manage an MPC problem using a CasADi dynamics model and symbolic cost
# using direct collocation to handle the dynamics

import casadi as ca
import numpy  as np
import matplotlib.pyplot as plt
from KinematicBicycleCar import KinematicBicycleCar

class MpcProblem:

    def __init__(self, params):

        self.model = KinematicBicycleCar(N=params["N"], step=params["step"])
        self.sys  = self.model.getDae()
        self.cost = 0
        self.Uk_prev = None
        self.indices_to_stop = None
        self.indices_to_start = None
        # hack

        simparams = {'x':self.sys.x[0], 'p':self.sys.u[0], 'ode':self.sys.ode[0]}
        self.sim = ca.integrator('F', 'idas', simparams, {'t0':0, 'tf':self.model.step})

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

    def bound_x(self, xc, yc, phi, dl, dr):
        # left bound by 2 points
        pl1 = np.array([xc + dl*np.cos(phi + np.pi/2.0),
                       yc + dl*np.sin(phi + np.pi/2.0)])
        pl2 = np.array([pl1[0]+np.cos(phi),
                        pl1[1]+np.sin(phi)])
        slopel = (pl2[1]-pl1[1])/(pl2[0]-pl1[0]); slopel = np.clip(slopel, -1e4, 1e4)
        offsetl = pl1[1] - pl1[0]*slopel

        # right bound
        pr1 = np.array([xc - dr*np.cos(phi + np.pi/2.0),
                       yc - dr*np.sin(phi + np.pi/2.0)])
        pr2 = np.array([pr1[0]+np.cos(phi),
                        pr1[1]+np.sin(phi)])
        sloper = (pr2[1]-pr1[1])/(pr2[0]-pr1[0]); sloper = np.clip(sloper, -1e4, 1e4)
        offsetr = pr1[1] - pr1[0]*sloper

        slopes = [(slopel, pl1, pl2), (sloper, pr1, pr2)]
        
        bounds = []
        # xc, yc should definitely be a feasible point
        
        for (slope, p, q) in slopes:
            offset = p[1]-p[0]*slope
            
            # We determine whether it is an upper or lower bound
            # by making sure center x,y is a feasible point
            
            if 0.0 <= xc*slope + yc*(-1.0) + offset and \
                      xc*slope + yc*(-1.0) + offset <= np.inf:
                bounds.append(np.array([np.inf, slope, -1.0, offset, 0.0]))
                
            elif -np.inf <= xc*slope + yc*(-1.0) + offset and \
                            xc*slope + yc*(-1.0) + offset <= 0.0:
                bounds.append(np.array([0.0, slope, -1.0, offset, -np.inf]))
            else:
                raise ValueError("HUGE ERROR at a, b, c =", slope, -1.0, offset, "\ncenter", center_x, center_y)
        return bounds



    def build_problem(self, ic:np.array, estimated_path:np.array, left_widths:np.array, right_widths:np.array, weights:dict):
        self.weights = weights
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
        fixed_points = [] # any hard constraints on a certain state

        # For plotting x and u given w
        x_plot = []
        u_plot = []
        self.x_center_plot = np.empty((self.model.n, self.model.N))

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
        self.steering_magnitude_cost = 0.0
        self.accel_magnitude_cost = 0.0

        # Create a matrix function
        # HACK: breaks if multiple state variables instead of one vector state
        # same for multiple control variables instead of one vector control!
        f = ca.Function('f', [self.sys.x[0], self.sys.u[0]],
                        [self.sys.ode[0], self.cost],
                        ['x', 'u'],['xdot', 'L'])

        # for plotting
        bounds = self.bound_x(estimated_path[0,0], estimated_path[1,0], estimated_path[3,0], left_widths[0], right_widths[0])

        # Formulate the NLP
        for k in range(self.model.N):
            # New NLP variable for the control
            Uk = ca.MX.sym("U_"+str(k), self.model.m)
            w.append(Uk)
            lbw.append(np.reshape(self.model.lowerbounds_u(k), (self.model.m,)))
            ubw.append(np.reshape(self.model.upperbounds_u(k), (self.model.m,)))
            # recall w0 is the guess of w, the decision variables
            w0.append(self.model.control_estimate[:,k])
            u_plot.append(Uk)

            # Add to control costs
            if None != self.Uk_prev:
                self.jerk_cost += (Uk[0]-self.Uk_prev[0])**2
                self.steering_change_cost += (Uk[1]-self.Uk_prev[1])**2
                self.steering_magnitude_cost += Uk[1]**2
                self.accel_magnitude_cost += Uk[0]**2

            self.Uk_prev = Uk

            # State at collocation points
            Xc = []
            xy = estimated_path[0:2,k]
            for j in range(self._d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), self.model.n)
                Xc.append(Xkj)
                w.append(Xkj)
                ub = np.reshape(self.model.upperbounds_x(k), (self.model.n,))
                ub[2] = estimated_path[2,k]*2.0 # limit to 2x desired speed
                lbw.append(np.reshape(self.model.lowerbounds_x(k), (self.model.n,)))
                ubw.append(ub)
                w0.append(self.model.state_estimate[:,k])                

                # Add the polygonal bounds at step k                
                bounds = self.bound_x(estimated_path[0,k], estimated_path[1,k], estimated_path[3,k], left_widths[k], right_widths[k])

                for (ub, a, b, c, lb) in bounds:
                    ubg.append(np.reshape(ub,(1,)))
                    lbg.append(np.reshape(lb,(1,)))
                    if a != 0:
                        g.append(Xkj[0]*a + Xkj[1]*b + c)
                    else:
                        g.append(Xkj[1]*b + c)
                

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
            lbw.append(np.reshape(self.model.lowerbounds_x(k+1), (self.model.n,)))
            ubw.append(np.reshape(self.model.upperbounds_x(k+1), (self.model.n,)))
    
            w0.append(self.model.state_estimate[:,k+1])
            x_plot.append(Xk)

            # Add the polygonal bounds at step k+1
            bounds = self.bound_x(estimated_path[0,k+1], estimated_path[1,k+1], estimated_path[3,k+1], left_widths[k+1], right_widths[k+1])
                
            for (ub, a, b, c, lb) in bounds:
                ubg.append(np.reshape(ub,(1,)))
                lbg.append(np.reshape(lb,(1,)))
                g.append(Xk[0]*a + Xk[1]*b + c)
                

            # Add equality constraint
            g.append(Xk_end-Xk)
            lbg.append(np.zeros((self.model.n,)))
            ubg.append(np.zeros((self.model.n,)))
            

            # Weakly attract state to middle of road
            # xy_k is (x,y,angle)
            xy_k = estimated_path[0:2,k+1]
            v_des = estimated_path[2,k]
            phi_k = estimated_path[3,k+1]
            self.x_center_plot[0:2,k] = xy_k
            self.x_center_plot[-1,k] = v_des
            # recall Xk[0] and Xk[1] are world frame x-y position
            # Xk[2] is velocity, Xk[3] is angle in world frame
            # we want to match the x-y position and road angle
            self.attractive_cost += ((Xk[0]-xy_k[0])**2 + \
                                     (Xk[1]-xy_k[1])**2 + \
                                     10*(Xk[2] - v_des)**2 + \
                                     (Xk[3]-phi_k)**2)

            if self.indices_to_stop is not None and k >= self.indices_to_stop:
                g.append(Xk[2])
                lbg.append(np.zeros(1))
                ubg.append(np.zeros(1))
            if self.indices_to_start is not None and k < self.indices_to_start:
                g.append(Xk[2])
                lbg.append(np.zeros(1))
                ubg.append(np.zeros(1))


        # This attracts the car to the middle of the road
        # Several papers make the steering change cost really big
        cost = self.weights["accuracy"]*self.attractive_cost + \
               self.weights["jerk"]*self.jerk_cost + \
               self.weights["steering change"]*180.0/np.pi*self.steering_change_cost + \
               self.weights["acceleration"]*self.accel_magnitude_cost# + \
               # 1.0*J # belongs to direct_collocation,  leftover
               # from the example this code was built from and unused
            


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
        lbg = np.concatenate(lbg) # yikes
        ubg = np.concatenate(ubg)

        # this is the format ipopt wants
        prob = {'f': cost, 'x': w, 'g': g}
        bounds = {'x0':w0, 'lbx':lbw, 'ubx':ubw, 'lbg':lbg, 'ubg':ubg}
        problem = (prob, bounds, w, x_plot, u_plot)
        return problem

    def solve(self, problem):
        prob, bounds, w, x_plot, u_plot = problem
        # Create an NLP solver
        solver = ca.nlpsol('solver', 'ipopt', prob, {'verbose':False});

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        # Solve the NLP
        #sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        sol = solver(**bounds)
        x_opt, u_opt = trajectories(sol['x'])
        self.x_opt = x_opt.full() # to numpy array
        self.u_opt = u_opt.full() # to numpy array

        # This ensures the control does not change drastically from the previous
        # (already-executed) control
        print("u_opt", [self.u_opt[0,0], self.u_opt[1,0]])
        self.Uk_prev = [self.u_opt[0,0], self.u_opt[1,0]]# we don't need this? we do.


        return self.x_opt, self.u_opt, solver.stats()

    def simulate(self, x:np.array, u:np.array):
      r = self.sim(x0=x, p=u)
      xf = r['xf']
      return np.reshape(xf, (self.model.n,))

