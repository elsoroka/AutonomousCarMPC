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

Segment = namedtuple('Segment', ['curve', 'start_pct', 'end_pct'])

class Roadrunner:

	def __init__(self, road_center:np.array, road_width:np.array,
		P=20, start_pct = 0.4, end_pct = 0.6):

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

		# Fit segments to the road points
		self.segments = []
		i = 0
		while i < n_points-P:
			curve = bezier.Curve.from_nodes( \
			    np.asfortranarray(np.transpose( \
				self.to_body_frame(road_center[i:i+P], self.angles[i])
				)))
			# so if P = 20 and end_pct = 60%, we get 12.
			end_idx = int(P*self.end_pct)
			start_idx = int(P*self.start_pct)

			if i == 0:
				start_xy = self.to_body_frame(self.road_center[i+start_idx:i+start_idx+1], self.angles[i], offset=self.road_center[i,:])
			else:
				start_xy = prev_end_xy
			end_xy   = self.to_body_frame(self.road_center[i+end_idx  :i+end_idx+1],   self.angles[i], offset=self.road_center[i,:])
			prev_end_xy = end_xy

			self.segments.append(Segment(curve=curve,
										 start_pct = curve.locate(np.asfortranarray(np.reshape(start_xy,(2,1)))),
										 end_pct   = curve.locate(np.asfortranarray(np.reshape(end_xy,  (2,1)))),
								))
			# Then we should start the next curve at i + (end_pt - start_pt)
			i += (end_idx - start_idx)

			print("Incremented i to", i)
			print("Added:\n", self.segments[-1])
		# Initialization
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
		offset_pct = offset_xy/self.segments[self._segment_ptr].curve.length
		return self.dist_traveled_xy + self.segments[self._segment_ptr].curve.evaluate(self.current_pct + offset_pct)


	def advance(self, step_xy=0.0):
		# curve length
		seg = self.segments[self._segment_ptr]
		step_pct = step_xy/seg.curve.length 
		# we are advancing along the path
		if step_pct > 0.0:
			while step_pct > (seg.end_pct - self.current_pct):
				step_pct -=  (seg.end_pct - self.current_pct)
				# Save the distance we traveled along this curve (k)
				self.dist_traveled_xy += seg.curve.length*(seg.end_pct - seg.start_pct)
				# Increment the segment_ptr, so we now use a new curve (k+1)
				self._segment_ptr += 1
				self.current_pct = self.segments[self._segment_ptr].start_pct
		#elif step < 0.0:
	#		while step < (self.current_pct - self.start_pct)
	# TODO: implement this
			#print("Incremented segment_ptr")

		self.current_pct += step_pct

		
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


	def reset(self)->None:
		'''Reset the roadrunner to the start of the road'''
		self._segment_ptr = 0
		self.dist_traveled_xy = 0.0
		self.current_pct = self.segments[self._segment_ptr].start_pct


	# PLOTTING
	def plot(self, ax=None, n_points=10):
		'''If given ax, draw on ax, else make a new plot.
		n_points: Number of points to plot.'''
		if ax is None:
			fig,ax = plt.subplots(1,1)

		s = np.linspace(self.start_pct, self.end_pct, n_points)

		for i,seg in enumerate(rr.segments):
			# Recall these points are fit in the car body frame
			xy = np.transpose(seg.curve.evaluate_multi(s))
			# Transform back to world frame
			xy = rr.to_world_frame(xy,rr.angles[i*4], offset=rr.road_center[i*4,:])
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
		new_center[:,0] = np.multiply(road_center[:,0]-offset[0], np.cos(angle)) + \
						  np.multiply(road_center[:,1]-offset[1], np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0]-offset[0], -np.sin(angle)) + \
						  np.multiply(road_center[:,1]-offset[1], np.cos(angle))
		return new_center

	@staticmethod
	def to_world_frame(road_center:np.array, angle:float, offset:np.array)->np.array:
		new_center = np.empty(np.shape(road_center))

		new_center[:,0] = np.multiply(road_center[:,0], np.cos(angle)) + \
						  np.multiply(road_center[:,1], -np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0], np.sin(angle)) + \
						  np.multiply(road_center[:,1], np.cos(angle))
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
	#print(rr.evaluate(0.0))

	#print(rr.segments[rr._segment_ptr].curve.length, "length")
	#rr.advance(rr.segments[rr._segment_ptr].curve.length*0.2)
	#print(rr.evaluate(0.0))

	import matplotlib.pyplot as plt
	fig,ax = plt.subplots(1,1)
	rr.plot(ax)


	plt.show()