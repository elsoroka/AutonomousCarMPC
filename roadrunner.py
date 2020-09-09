#
# DON'T USE THIS FILE, USE ROADRUNNER_2.py THIS FILE DOES NOT WORK.
# 09/08/2020
#
# A class to handle feeding the road splines to the controller
# File created: 08/08/2020
# Emiko Soroka

import numpy as np
import bezier
N_POINTS = 20

from collections import namedtuple, deque

class RoadSegment():
	
	Bound = namedtuple('Bound', ['slope', 'offset', 'side'])
	
	def __init__(self, road_center:np.array, road_width:np.array,
		         start=0.4, end=0.6):
		'''road_center: 2'''
		self.road_center = road_center
		self.road_width  = road_width
		self.start       = start
		self.end         = end

		self.P, = np.shape(self.road_width)
		assert (self.P,2) == np.shape(self.road_center)

		# Start and end angles
		# note the last point is used to get the angle at the second-to-last point
		# so the road segment is actually from the first point to the second-to-last point.
		a1 = np.arctan2(road_center[1,1]-road_center[0,1], road_center[1,0]-road_center[0,0])
		a2 = np.arctan2(road_center[-1,1]-road_center[-2,1], road_center[-1,0]-road_center[-2,0])

		assert np.abs(a1) <= np.pi/180.0 or np.abs(a1 - 2*np.pi) <= np.pi/180.0
		# The start angle should be very close to 0
		
		# We draw the cut line that separates this road segment from the next
		slope2 = np.tan(a2 + np.pi/2)
		slope2 = np.sign(slope2)*1e4 if slope2 > 1e4 else slope2 # limit

		self.bound = \
			self.Bound(slope=slope2, offset=np.reshape(road_center[-1,:],2), side=">")

		# Now we fit a curve to the points
		nodes = np.asfortranarray(np.transpose(road_center))
		self.curve = bezier.Curve.from_nodes(nodes)

		self.dist_traveled = 0.0


	@staticmethod
	def check_bound(bound,p):
		#print("Bound:\n", bound)
		p = np.reshape(p,2)

		# Catch the case where the bound has a vertical line?
		if bound.side == ">" and np.sign(bound.slope) == 1 \
		or bound.side == "<" and np.sign(bound.slope) == -1:
			res = p[1] >= bound.slope*(p[0]-bound.offset[0]) + bound.offset[1]
			return res
		else:
			res =  p[1] < bound.slope*(p[0]-bound.offset[0]) + bound.offset[1]
			return res

	def dist_to_center(self, p):
		p = np.reshape(p,2)
		points = np.linspace(0,1,self.P)
		points = self.curve.evaluate_multi(points)
		best = np.Inf
		best_idx = 0
		for i in range(self.P):
			angle = np.arctan2(p[1]-points[1,i], p[0]-points[0,i])
			if np.abs(angle) - np.pi/2 < best:
				best = np.abs(angle)-np.pi/2
				best_idx = i
			#print("i, dist(i):", np.sqrt((p[0]-points[0,i])**2 + (p[1]-points[1,i])**2))

		dist = np.sqrt((p[0]-points[0,best_idx])**2 + (p[1]-points[1,best_idx])**2)
		print("best distance:", dist)
		return dist, best_idx

	def __contains__(self, p:np.array):
		dist, idx = self.dist_to_center(p)
		p = np.reshape(p,2)
		print("p[0], road_center[0,0]", p[0], self.road_center[0,0])
		print("dist", dist, "width", self.road_width[idx])
		print("bound", self.check_bound(self.bound, p))
		res = self.check_bound(self.bound, p) and \
		      p[0] >= self.road_center[0,0] and \
		      dist <= self.road_width[idx]
		return res


class Roadrunner():

	def __init__(self, road_center:np.array, road_width:np.array, P=20):

		self.P = P

		# TODO: check proper sizes of road_center (n_points x 2) and width (n_points x 1)
		self.road_center = road_center
		n_points,_ = np.shape(road_center)
		self.road_width = road_width
		
		self.angle = np.empty(np.shape(road_width))
		for i in range(n_points-1):
			# arctan2 covers the whole unit circle
			# range is -pi to +pi.
			self.angle[i] = np.arctan2(
				(self.road_center[i+1,1]-self.road_center[i,1]),
				(self.road_center[i+1,0]-self.road_center[i,0]))

		# Fill in the last one based on the second-to-last.
		self.angle[-1] = self.angle[-2]

		self.segments = [RoadSegment( \
				self.to_body_frame(road_center[i:i+P], self.angle[i]),
				road_width[i:i+P]) for i in range(n_points-P)]

		self.segment_ptr = 0


	def advance(self, step:float)->RoadSegment:
		# Imagine the road as a set of vectors
		# which are the road center-points with direction pointing to the next point.
		# We want to draw the perpendicular line and
		# determine if we are before that line or after it.

		seg = self.segments[self.segment_ptr]
		seg.dist_traveled += step
		# We have finished traversing this curve:
		while seg.curve.length/self.P < seg.dist_traveled:
			step = seg.dist_traveled - seg.curve.length/self.P
			self.segment_ptr += 1
			#print("ptr is ", self.segment_ptr)
			if self.segment_ptr == len(self.segments):
				raise StopIteration
				
			seg = self.segments[self.segment_ptr]
			seg.dist_traveled += step

		return self.segments[self.segment_ptr]

	def evaluate(self, s=None)->np.array:
		# s can be float or np.array or if none, evaluate at current point
		pts = None

		if s is None:
			s = float(self.segments[self.segment_ptr].dist_traveled/self.segments[self.segment_ptr].curve.length)
		# test
		pts = np.zeros((2,np.size(s)))
		section = self.segments[self.segment_ptr:self.segment_ptr+2]
		for seg in section:
			if type(s) == float:
				pts += seg.curve.evaluate(s)
			else:
				pts += seg.curve.evaluate_multi(s)
		pts /= len(section)

		
		pts = np.transpose(pts)
		return self.to_world_frame(pts, self.angle[self.segment_ptr], self.road_center[self.segment_ptr,:])

	def get_width(self)->float:
		return self.road_width[self.segment_ptr]
	
	def get_angle(self)->float:
		return self.angle[self.segment_ptr]

	def get_segment(self)->RoadSegment:
		return self.segments[self.segment_ptr]

	def reset(self)->None:
		self.segment_ptr = 0
		for s in self.segments:
			s.dist_traveled = 0.0

	def evaluate_at(self, dist)->np.array:
		seg = self.advance(dist)
		return self.evaluate(float(seg.dist_traveled/seg.curve.length))


	@staticmethod
	def to_body_frame(road_center:np.array, angle:float)->np.array:
		new_center = np.empty(np.shape(road_center))
		new_center[:,0] = np.multiply(road_center[:,0]-road_center[0,0], np.cos(angle)) + \
						  np.multiply(road_center[:,1]-road_center[0,1], np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0]-road_center[0,0], -np.sin(angle)) + \
						  np.multiply(road_center[:,1]-road_center[0,1], np.cos(angle))
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

	from road import test_road
	(N_POINTS,_) = np.shape(test_road)
	test_width = np.ones(N_POINTS)*3.0

	rr = Roadrunner(test_road, test_width, P = 20)
	import matplotlib.pyplot as plt


	fig, ax = plt.subplots(1,1)

	rr.reset()

	test_points = np.empty((100,2))
	for i in range(100):
		seg = rr.advance(1)
		pts = rr.evaluate(seg.dist_traveled/seg.curve.length)
		
		test_points[i,:] = pts

	
	plt.scatter(test_road[:,0], test_road[:,1])
	plt.plot(test_points[:,0], test_points[:,1])
	plt.scatter(test_points[:,0], test_points[:,1])

	plt.show()
