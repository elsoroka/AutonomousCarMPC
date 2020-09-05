# A class to handle feeding the road splines to the controller
# File created: 08/30/2020
# Emiko Soroka
# the previous one was terrible and riddled with bugs and didn't work.

# Goals
# have a current point along the road
# be able to lookahead a given distance forward/back from the current point
# advance the current point (separately from moving)
# loop road points / queue
# use center of each curve fit, not first piece

import numpy             as np
import matplotlib.pyplot as plt
import bezier
from collections import namedtuple

Segment = namedtuple('Segment', ['curve', 'start_pct', 'end_pct', 'transform_angle', 'transform_offset'])

class Roadrunner:

	def __init__(self, road_center:np.array, road_width:np.array,
		P=10, start_pct = 0.3, end_pct = 0.7):

		self.P = P # Number of points to fit at one time
		self.start_pct = start_pct # Percentage length to start using the curve
		# so if the curve goes from 0 to 1, you start using it at 0.4
		self.end_pct = end_pct # and switch to the new curve at 0.6

		# TODO: check proper sizes of road_center (n_points x 2) and width (n_points x 1)
		self.road_center = road_center
		n_points,_ = np.shape(road_center)
		self.road_width = road_width
		
		self.angles = np.empty(np.shape(road_width))
		for i in range(n_points-1):
			# arctan2 covers the whole unit circle
			# range is -pi to +pi.
			self.angles[i] = np.arctan2(
				(self.road_center[i+1,1]-self.road_center[i,1]),
				(self.road_center[i+1,0]-self.road_center[i,0]))

		# Fill in the last angle based on the second-to-last.
		self.angles[-1] = self.angles[-2]
		

		self.segments = []
		# Fit segments to the road points
		i = 0
		# so if P = 20 and end_pct = 60%, we get 12.
		end_idx = int(P*self.end_pct)
		start_idx = int(P*self.start_pct)

		while i < n_points-P:
			angle  = self.angles[i]
			offset = self.road_center[i]
			curve  = bezier.Curve.from_nodes( \
			             np.asfortranarray(np.transpose( \
			    	         self.to_body_frame(road_center[i:i+P], angle))
			         ))

			start_xy = self.to_body_frame(self.road_center[i+start_idx:i+start_idx+1], angle, offset)
			end_xy   = self.to_body_frame(self.road_center[i+end_idx  :i+end_idx+1],   angle, offset)

			start_pct,_ = Roadrunner.find_closest_pt(curve, np.reshape(start_xy,(2,1)))
			end_pct,_   = Roadrunner.find_closest_pt(curve, np.reshape(end_xy,  (2,1)))
			self.segments.append(Segment(curve     = curve,
										 start_pct = start_pct,
										 end_pct   = end_pct,
										 transform_angle  = angle,
										 transform_offset = offset
								))
			# Then we should start the next curve at i + (end_pt - start_pt)
			i += (end_idx - start_idx)


		# Set distances traveled to 0.
		self.reset()

		# OK so we want to store the start/end point on the curve
		# using locate(x,y) to find it
		# so get point xy_0.4 as road_center[int(P*0.4)]
		# and then get point xy_0.6 as road_center[int(p*0.6)]
		# Then we know how to fit the next curve.


	def evaluate(self, offset_xy=0.0):
		'''Given an offset (in meters) from self.current_position,
		evaluate the curve at that given point.
		The offset may be negative.
		'''
		# o o  o o  o  o o  o o 
		# -----------------
		#     0.4  0.6 
		# -----------------
		# scale to percentage of curve length
		# If this offset will put us on another curve,
		# save our state so we can get back to the right place
		seg = self.segments[self._segment_ptr]
		offset_pct = offset_xy/seg.curve.length
		result = None
		state = None
		if (np.sign(offset_pct) ==  1 and  offset_pct >= seg.end_pct - self.current_pct) or \
		   (np.sign(offset_pct) == -1 and -offset_pct >= self.current_pct - seg.start_pct):
			# save current state
		   state = self.save_state()
		   self.advance(offset_xy)

		   # reset since we saved the state
		   offset_xy = 0.0; offset_pct = 0.0
		   seg = self.segments[self._segment_ptr]
		
		result = seg.curve.evaluate(self.current_pct + offset_pct)
		result = self.to_world_frame(np.reshape(result,(1,2)), \
									 angle=seg.transform_angle,
								     offset=seg.transform_offset)
		# restore current state
		if state is not None:
			self.reset(**state)

		return result


	def advance(self, step_xy=0.0):
		# curve length
		seg = self.segments[self._segment_ptr]

		# Convert the step in meters to a step as a percentage
		# of the current curve fit.
		step_pct = step_xy/seg.curve.length 
		

		if step_pct > 0.0:
			# While the step is larger than the rest of the curve
			while step_pct > (seg.end_pct - self.current_pct):

				# Advance to the next curve:
				step_pct  -= (seg.end_pct - self.current_pct)

				# Save the distance we traveled along this curve (k)
				self.dist_traveled_xy += seg.curve.length*(seg.end_pct - self.current_pct)
				# Increment the segment_ptr, so we now use a new curve (k+1)
				self._segment_ptr += 1
				seg = self.segments[self._segment_ptr]
				if self._segment_ptr < 0 or self._segment_ptr >= len(self.segments):
					print("WARNING: you have run out of road points!")

				# Now we're at the beginning of the new curve
				self.current_pct = seg.start_pct

		elif step_pct < 0.0:
			step_pct = np.abs(step_pct) # now it's positive and easier to work with
			while step_pct > (self.current_pct - seg.start_pct):
				step_pct  -= (self.current_pct - seg.start_pct)

				self.dist_traveled_xy += seg.curve.length*(self.current_pct - seg.start_pct)
				self._segment_ptr -= 1
				seg = self.segments[self._segment_ptr]

				if self._segment_ptr < 0 or self._segment_ptr >= len(self.segments):
					print("WARNING: you have run out of road points!")
				self.current_pct = seg.end_pct

			step_pct *= -1 # reset sign so the remainder is correctly negative
		
		# Add any leftover step after we advanced the segment_ptr.
		self.current_pct += step_pct
		#print("Advanced to", self.current_pct)
		# Done!

		
	def get_width(self)->float:
		# TODO: Fix. This is broken, there are fewer segments
		# than there are road points.
		# it has been de-prioritized since we're using constant width
		# roads to test.
		return self.road_width[self._segment_ptr]
	
	def get_angle(self)->float:
		# Evaluate the tangent vector to the curve at our current point
		curve = self.segments[self._segment_ptr].curve
		tangent_vector = curve.evaluate_hodograph(self.current_pct)
		# Now we can get a precise angle
		return np.arctan2(tangent_vector[1], tangent_vector[0])


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

			dist = np.linalg.norm(curve.evaluate(x_i) - match_pt,2)
			if dist < min_dist:
				min_dist = dist
				min_idx = idx
		return min_idx, dist

	@staticmethod
	def find_closest_pt(curve, match_pt:np.array, runs=3)->(float, float):
		'''Given an xy point match_pt, find the closest point on the curve B.
		Returns the corresponding s for the curve B(s) and the distance
		between match_pt and B(s) (which can be 0 if match_pt is on B).
		Not guaranteed to work if match_pt is not "close" to B.'''
		
		# Will be the correct value if match_pt is on the curve
		min_s, dist = curve.locate(match_pt), 0.0
		if min_s is None: # otherwise it's None

			start = 0.0; end = 1.0; # start by searching the whole interval
			x = None
			for r in range(runs):
				x = np.linspace(start,end,10)
				min_idx,dist = Roadrunner._find_closest_in_x_to_pt(curve, x, match_pt)
				min_s = x[min_idx]

				start = np.max([0.0, x[min_idx]-1/(10*(r+1))])
				end = np.min([1.0, x[min_idx]+1/(10*(r+1))])

			# Check: if this closest point is "too far"
			if dist > 2.5:
				print("find_closest_pt may be unreliable: closest match found for {} is {} away at s = {}, B(s) = {}".format(match_pt, dist, min_s, curve.evaluate(min_s)))

			return min_s, dist


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
			xy = rr.to_world_frame(xy,seg.transform_angle, seg.transform_offset)
			# Plot the points
			ax.plot(xy[:,0], xy[:,1])

		return ax


	# Transformations between body and world frame

	@staticmethod
	def to_body_frame(road_center:np.array, angle:float, offset=None)->np.array:
		
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
		new_center = np.empty(np.shape(road_center))

		new_center[:,0] = np.multiply(road_center[:,0],  np.cos(angle)) + \
						  np.multiply(road_center[:,1], -np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0],  np.sin(angle)) + \
						  np.multiply(road_center[:,1],  np.cos(angle))
		new_center[:,0] += offset[0]
		new_center[:,1] += offset[1]
		return new_center


if __name__ == "__main__":
	import unittest
	import numpy as np
	from road import test_road
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