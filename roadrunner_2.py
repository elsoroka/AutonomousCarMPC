# A class to handle feeding the road splines to the controller
# File created: 08/30/2020
# Emiko Soroka
# the previous one was riddled with bugs and didn't work.

# How it works:
# We split the road into segments.
#   Each segment is a Bezier curve made by fitting "P" points to each curve.
#   It seems the middle of the curve fits better than the two ends
#   (especially around tight turns),
#   so we use each curve between its start_pct (percentage of length,
#   for example start at 20% of the curve) and end_pct.
#   This means successive curves overlap.
#
# We track a current position on the road.
#   We have a segment_ptr to keep track of which curve we're on,
#   and a current_pct to keep track of how far along the curve we are.
#   When we get past the curve's end_pct, we switch to the next curve.
#
# We can look ahead or behind the current position by a specified distance
#   (in meters) without losing our current position.
#   So to look ahead for the k-th step, figure out where you expect to be
#   (using previously computed velocity, state, etc.) and look ahead that distance.
#
# We can also advance the current position by some distance (in meters).
#   So if you execute an MPC step where you travel 0.25 m
#   call roadrunner.advance(0.25) to move your current_position.

import bezier
import numpy             as np
import matplotlib.pyplot as plt
from   collections       import namedtuple

# A road Segment has a Bezier curve fit from some body-frame points
# a start_pct and end_pct (governs when to start and stop using this curve)
# and the x-y point and angle that transform
# the x-y results of curve.evaluate() from body frame to world frame.
Segment = namedtuple('Segment', ['curve', 'width', 'start_pct', 'end_pct', 'transform_angle', 'transform_offset'])

class OutOfRoadPointsException(Exception):
	pass

class Roadrunner:

	def __init__(self, road_center:np.array, road_width:np.array,
		P=20, start_pct = 0.35, end_pct = 0.65):

		self.P = P # Number of points to fit at one time
		# Fraction of curve length where we start using the curve
		# so if the curve goes from 0 to 1, we start using it at start_pct (0.3)
		self.start_pct = start_pct
		 # and switch to the new curve at end_pct (0.7)
		self.end_pct = end_pct
		# Yes, percentage should be 0 - 100, not 0 - 1.
		# The variables are misnamed. :(

		# TODO: check proper sizes of road_center (n_points x 2) and width (n_points x 1)
		self.road_center = road_center
		n_points,_       = np.shape(road_center)
		self.road_width  = road_width
		
		# Fit segments to the road points
		self.segments = []
		self.widths   = []
		i = 0
		# so if P = 20 and end_pct = 60%, we get 12.
		end_idx   = int(P*self.end_pct)
		start_idx = int(P*self.start_pct)

		while i < n_points-P:
			# First: get the angle and offset to transform the road points
			# from world frame to car body frame

			# arctan2 covers the whole unit circle; range is -pi to +pi.
			angle = np.arctan2((self.road_center[i+1,1]-self.road_center[i,1]),
							   (self.road_center[i+1,0]-self.road_center[i,0]))
			#angle = self.angles[i]
			offset = self.road_center[i]
			# Now transform the points and fit a curve to them.
			road_body_frame = self.to_body_frame(road_center[i:i+P], angle)
			curve  = bezier.Curve.from_nodes(np.asfortranarray( \
					 	np.transpose(road_body_frame)
			         ))
			distances = np.empty(P)
			for j in range(P):
				tmp = Roadrunner.find_closest_pt(curve, np.reshape(road_body_frame[j],(2,1)), runs=4, start=0,end=1)
				distances[j] = tmp[0]

			curve_width = bezier.Curve.from_nodes(np.asfortranarray( \
					np.vstack([distances, road_width[i:i+P]])
				))
			
			# Now we have a start_pct and end_pct (default: 0.3 to 0.7)
			# but not all the curves will overlap perfectly
			# so we can refine it a bit
			# and make sure each successive curve picks up
			# right after the last one leaves off.
			# Figure out where each curve's start_pct and end_pct is
			# in x-y coordinates
			start_xy = self.to_body_frame(np.reshape(self.road_center[i+start_idx],(1,2)), angle, offset)
			end_xy   = self.to_body_frame(np.reshape(self.road_center[i+end_idx],  (1,2)),   angle, offset)

			start_pct,_ = Roadrunner.find_closest_pt(curve, np.reshape(start_xy,(2,1)), runs=4, start=0,end=0.5)
			end_pct,_   = Roadrunner.find_closest_pt(curve, np.reshape(end_xy,  (2,1)), runs=4, start=0.5,end=1)
			self.segments.append(Segment(curve     = curve,
										 width     = curve_width,
										 start_pct = start_pct,
										 end_pct   = end_pct,
										 transform_angle  = angle,
										 transform_offset = offset
								))
			# Then we should start the next curve at i + (end_pt - start_pt)
			i += (end_idx - start_idx)


		# Set distances traveled to 0.
		self.reset()


	def evaluate(self, offset_xy=0.0, full_data=False):
		'''Given an offset (in meters) from self.current_position,
		evaluate the curve at that given point.
		The offset may be negative.
		'''
		# scale to percentage of curve length
		seg = self.segments[self._segment_ptr]
		offset_pct = offset_xy/seg.curve.length

		result = None
		state = None
		
		# If this offset will put us on another curve,
		# save our state so we can get back to the right place
		if (np.sign(offset_pct) ==  1 and  offset_pct >= seg.end_pct - self.current_pct) or \
		   (np.sign(offset_pct) == -1 and -offset_pct >= self.current_pct - seg.start_pct):
		   state = self.save_state()
		   # advance() takes care of moving the segment_ptr and current_pct
		   self.advance(offset_xy)

		   # we've advanced to the correct place, now offset is 0.0
		   offset_xy = 0.0; offset_pct = 0.0
		   seg = self.segments[self._segment_ptr]
		
		# evaluate and transform result to world frame
		result = seg.curve.evaluate(self.current_pct + offset_pct)
		result = self.to_world_frame(np.reshape(result,(1,2)), \
									 angle  = seg.transform_angle,
								     offset = seg.transform_offset)
		if full_data:
			result = (result, self.get_angle(), self.get_width())

		# restore current state
		if state is not None:
			self.reset(**state)

		return result


	def advance(self, step_xy=0.0):
		'''Move current_pct along the segments by a distance
		step_xy (in meters)'''

		seg = self.segments[self._segment_ptr]

		# Convert the step in meters to a step as a percentage
		# of the current curve fit.
		step_pct = step_xy/seg.curve.length 
		

		if step_pct > 0.0:
			# While the step is larger than the rest of the curve
			while step_pct > (seg.end_pct - self.current_pct):

				# Save the distance we traveled along this curve (k)
				self.dist_traveled_xy += seg.curve.length*(seg.end_pct - self.current_pct)
				# Increment the segment_ptr, so we now use a new curve (k+1)
				self._segment_ptr += 1

				if self._segment_ptr < 0 or self._segment_ptr >= len(self.segments):
					raise OutOfRoadPointsException("Ran out of road points (looking forward)!")


				# Advance to the next curve:
				step_xy  -= seg.curve.length*(seg.end_pct - self.current_pct)
				seg = self.segments[self._segment_ptr]
				# Now we're at the beginning of the new curve
				self.current_pct = seg.start_pct
				step_pct = step_xy/seg.curve.length


		elif step_pct < 0.0:
			step_pct = np.abs(step_pct) # now it's positive and easier to work with
			step_xy  = np.abs(step_xy)
			while step_pct > (self.current_pct - seg.start_pct):
				
				# Save the (negative!) distance we traveled along this curve
				self.dist_traveled_xy -= seg.curve.length*(self.current_pct - seg.start_pct)
				# Move to the previous curve
				self._segment_ptr -= 1

				if self._segment_ptr < 0 or self._segment_ptr >= len(self.segments):
					raise OutOfRoadPointsException("Ran out of road points (looking backward)!")
				

				# Advance to the next curve
				step_xy  -= seg.curve.length*(self.current_pct - seg.start_pct)
				seg = self.segments[self._segment_ptr]
				# Now we're at the end of the new curve
				self.current_pct = seg.end_pct
				step_pct = step_xy/seg.curve.length

			step_pct *= -1 # reset sign so the remainder is correctly negative
			step_xy  *= -1

		# Add any leftover step after we advanced the segment_ptr.
		self.current_pct += step_pct
		#print("Advanced to", self.current_pct)
		# Done!


	def advance_s(self, delta_s)->float:
		'''Advance by a distance-along-the-curve s,
		not an x-y distance.'''
		seg = self.segments[self._segment_ptr]
		if self.current_pct + delta_s > seg.end_pct:
			delta_xy = (seg.end_pct - self.current_pct)*seg.curve.length + \
			(delta_s - (seg.end_pct - self.current_pct))*self.segments[self._segment_ptr+1].curve.length
		else:
			delta_xy = delta_s*seg.curve.length
		self.advance(delta_xy)
		return delta_xy

	def advance_xy(self, xy)->float:
		'''Advance to a "close by" x-y point by finding the closest point
		to xy on the curve.'''
		seg = self.segments[self._segment_ptr]
		pt = np.asfortranarray(
			self.to_body_frame(np.reshape(xy,(1,2)), seg.transform_angle, seg.transform_offset))

		(s_new,_) = self.find_closest_pt(seg.curve,
										 np.reshape(pt,(2,1)),
										 start=self.current_pct,
										 end=self.current_pct + 0.2)
		# Advance to s_new, the closest point on the curve.
		delta_s = s_new - self.current_pct
		step_xy = self.advance_s(delta_s)
		return step_xy

		
	def get_width(self)->float:
		return self.segments[self._segment_ptr].width.evaluate(self.current_pct)[1]
		
	
	def get_angle(self)->float:
		# Evaluate the tangent vector to the curve at our current point
		curve = self.segments[self._segment_ptr].curve
		tangent_vector = curve.evaluate_hodograph(self.current_pct)
		# Now we can get a precise angle
		return np.arctan2(tangent_vector[1], tangent_vector[0]) + self.segments[self._segment_ptr].transform_angle


	def reset(self, segment_ptr=0, dist_traveled_xy=0.0, current_pct_offset=0.0)->None:
		'''Reset the roadrunner to the start of the road (default)
		or to a specified location.'''
		self._segment_ptr = segment_ptr
		self.dist_traveled_xy = dist_traveled_xy
		self.current_pct = self.segments[self._segment_ptr].start_pct + current_pct_offset

	def save_state(self)->dict:
		return {"segment_ptr"        : self._segment_ptr,
				"dist_traveled_xy"   : self.dist_traveled_xy,
				"current_pct_offset" : self.current_pct - self.segments[self._segment_ptr].start_pct}


	@staticmethod
	def _find_closest_in_x_to_pt(curve, x, match_pt):
		# We're looking for the closest point on the curve
		# to some point match_pt that is NOT on the curve
		min_dist = np.inf
		min_idx = 0
		for idx, x_i in enumerate(x):
			xy = curve.evaluate(x_i)
			dist = np.sqrt((xy[0]-match_pt[0])**2 + (xy[1]-match_pt[1])**2)
			if dist < min_dist:
				min_dist = dist
				min_idx = idx
		return min_idx, dist

	@staticmethod
	def find_closest_pt(curve, match_pt:np.array, runs=2, start=0.0, end=1.0)->(float, float):
		'''Given an xy point match_pt, find the closest point on the curve B.
		Returns the corresponding s for the curve B(s) and the distance
		between match_pt and B(s) (which can be 0 if match_pt is on B).
		Not guaranteed to work if match_pt is not "close" to B.'''
		
		# Will be the correct value if match_pt is on the curve
		min_s, dist = curve.locate(match_pt), 0.0
		if min_s is None: # otherwise it's None

			x = None
			for r in range(runs):
				x = np.linspace(start,end,256)
				min_idx,dist = Roadrunner._find_closest_in_x_to_pt(curve, x, match_pt)
				min_s = x[min_idx]

				# Here, we adjust the interval to search in
				# by making it 1/2 the size of the previous interval
				# and centered on the best point we found.
				start = np.max([0.0, x[min_idx]-0.25*(end-start)])
				end = np.min([1.0, x[min_idx]+0.25*(end-start)])

			# Check: if this closest point is "too far"
			if dist > 2.5:
				print("WARNING: find_closest_pt may be unreliable: closest match found for {} is {} away at s = {}, B(s) = {}".format(match_pt, dist, min_s, curve.evaluate(min_s)))

		return min_s, dist




	def center(self, step, k:int, desired_speed:callable)->np.array:

	    (xy, angle, _) = self.evaluate(step*np.sum([desired_speed(i) for i in range(k)]), full_data=True)

	    center = np.empty((3,))
	    center[0:2] = np.reshape(xy, 2)
	    center[2]   = angle
	    return center

	# Test

	def bound_x(self, step, k:int, desired_speed:callable)->[np.array]:
	    # Returns a 4-sided polygon bound, like this:
	    # x2-------x3
	    #   \  o  /
	    #  x1-----x4
	    # successive bounds will overlap each other
	    # so there is freedom to slow down or speed up the vehicle.
	    dist = step*np.sum([desired_speed(i) for i in range(k)])

	    dist_behind = dist - step*sum([desired_speed(i) for i in range(k-10,k)]) # 5 steps behind * timestep * velocity at point k
	    #print("Looking behind by", dist_behind, dist_behind - dist)
	    (center_minus, angle_minus, width_minus) = self.evaluate(dist_behind, full_data=True)
	    dist_ahead = dist + step*sum([desired_speed(i) for i in range(k+1,k+11)]) # 5 steps ahead * timestep * velocity at point k
	    #print("Looking ahead by", dist_ahead, dist_ahead - dist)
	    (center_plus, angle_plus, width_plus) = self.evaluate(dist_ahead, full_data=True)

	    center_minus = np.reshape(center_minus,(2,))
	    center_plus = np.reshape(center_plus,(2,))
	    
	    # we either have 0.0 >= ax + by + c >= -Inf
	    # or             Inf >= ax + by + c >= 0.0
	    
	    # since y = (x-x0)*slope + y0
	    # use the line 0 = x*slope - y + y0 - x0*slope as the upper / lower bound
	    
	    #            (upper)
	    #     p2-----slope23------p3
	    #    /                    /
	    # slope12 (upper)   slope34 (lower)
	    #  /                   /
	    # p1-----slope14-----p4
	    #        (lower)
	    
	    p1 = np.reshape(np.array([center_minus[0] + width_minus/2.0*np.cos(angle_minus-np.pi/2),
	                   center_minus[1] + width_minus/2.0*np.sin(angle_minus-np.pi/2)]), (2,))
	    
	    p2 = np.reshape(np.array([center_minus[0] + width_minus/2.0*np.cos(angle_minus+np.pi/2),
	                   center_minus[1] + width_minus/2.0*np.sin(angle_minus+np.pi/2)]), (2,))
	    
	    p3 = np.reshape(np.array([center_plus[0] + width_plus/2.0*np.cos(angle_plus+np.pi/2),
	                   center_plus[1] + width_plus/2.0*np.sin(angle_plus+np.pi/2)]), (2,))
	    
	    p4 = np.reshape(np.array([center_plus[0] + width_plus/2.0*np.cos(angle_plus-np.pi/2),
	                   center_plus[1] + width_plus/2.0*np.sin(angle_plus-np.pi/2)]), (2,))
	    
	    # This point is approximately the center of the bound
	    # It should definitely be a feasible point
	    center_x, center_y = 0.5*(center_minus + center_plus)
	    
	    
	    # The slopes between the points
	    slope12 = (p2[1]-p1[1])/(p2[0]-p1[0]); slope12 = 1e4 if np.isinf(slope12) else slope12
	    slope23 = (p3[1]-p2[1])/(p3[0]-p2[0]); slope23 = 1e4 if np.isinf(slope23) else slope23
	    slope43 = (p3[1]-p4[1])/(p3[0]-p4[0]); slope43 = 1e4 if np.isinf(slope43) else slope43
	    slope14 = (p1[1]-p4[1])/(p1[0]-p4[0]); slope14 = 1e4 if np.isinf(slope14) else slope14
	    
	    slopes = [(slope12, p1, p2), (slope23, p2, p3), (slope43, p4, p3), (slope14, p4, p1)]
	    
	    bounds = []
	    for (slope, p, q) in slopes:
	        offset = p[1]-p[0]*slope
	        
	        # We determine whether it is an upper or lower bound
	        # by making sure center x,y is a feasible point
	        
	        if 0.0 <= center_x*slope + center_y*(-1.0) + offset and \
	                  center_x*slope + center_y*(-1.0) + offset <= np.inf:
	            bounds.append(np.array([np.inf, slope, -1.0, offset, 0.0]))
	            
	        elif -np.inf <= center_x*slope + center_y*(-1.0) + offset and \
	                        center_x*slope + center_y*(-1.0) + offset <= 0.0:
	            bounds.append(np.array([0.0, slope, -1.0, offset, -np.inf]))
	        else:
	            raise ValueError("HUGE ERROR at ", k, "a, b, c =", slope, -1.0, offset, "\ncenter", center_x, center_y)
	    

	    return bounds, np.vstack([p1,p2,p3,p4])





	# PLOTTING
	def plot(self, ax=None, n_points=10):
		'''If given ax, draw on ax, else make a new plot.
		n_points: Number of points to plot.'''
		if ax is None:
			fig,ax = plt.subplots(1,1)

		s = np.linspace(self.start_pct, self.end_pct, n_points)
		

		for seg in self.segments:
			# Recall these points are fit in the car body frame
			xy = np.transpose(seg.curve.evaluate_multi(s))
			# Transform back to world frame
			xy = self.to_world_frame(xy,seg.transform_angle, seg.transform_offset)
			# Plot the points
			ax.plot(xy[:,0], xy[:,1])

		return ax


	# Transformations between body and world frame

	@staticmethod
	def to_body_frame(road_center:np.array, angle:float, offset=None)->np.array:
		'''Inverse of to_world_frame. Transform points to body frame
		using given angle and offset.'''
		# If no offset explicitly provided, default to the first point
		# so the first point will be transformed to [0,0]
		if offset is None:
			offset = np.reshape(road_center[0,:],2)
		
		new_center = np.empty(np.shape(road_center))
		new_center[:,0] = np.multiply(road_center[:,0]-offset[0],  np.cos(angle)) + \
						  np.multiply(road_center[:,1]-offset[1],  np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0]-offset[0], -np.sin(angle)) + \
						  np.multiply(road_center[:,1]-offset[1],  np.cos(angle))
		return new_center

	@staticmethod
	def to_world_frame(road_center:np.array, angle:float, offset:np.array)->np.array:
		'''Inverse of to_body_frame. Transform points to world frame
		using given angle and offset.'''
		new_center = np.empty(np.shape(road_center))

		new_center[:,0] = np.multiply(road_center[:,0],  np.cos(angle)) + \
						  np.multiply(road_center[:,1], -np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0],  np.sin(angle)) + \
						  np.multiply(road_center[:,1],  np.cos(angle))
		new_center[:,0] += offset[0]
		new_center[:,1] += offset[1]
		return new_center


# Small test to demonstrate code
if __name__ == "__main__":
	import unittest
	import numpy as np
	from   road  import test_road

	(n_points,_) = np.shape(test_road)
	test_width = 5.0*np.ones(n_points)
	rr = Roadrunner(test_road, test_width)

	import matplotlib.pyplot as plt

	print("Generating self-test plot...")
	
	fig,ax = plt.subplots(1,1)
	rr.plot(ax)
	rr.reset(segment_ptr=5) # approximately the middle of the road points
	sgn = 1
	points = np.empty((15,2))

	# Test evaluating points AHEAD of our current point
	for k in range(15):
		xy = rr.evaluate(k*5)	
		points[k,:] = np.reshape(xy, (1,2))

	ax.scatter(points[:,0], points[:,1], color="blue", label="ahead of current point")

	# Test evaluating points BEHIND our current point
	for k in range(15):
		xy = rr.evaluate(-k*5)	
		points[k,:] = np.reshape(xy, (1,2))

	ax.scatter(points[:,0], points[:,1], color="red", label="behind current point")

	plt.legend()
	plt.show()