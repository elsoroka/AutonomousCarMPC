# A class to handle feeding the road splines to the controller
# File created: 08/08/2020
# Emiko Soroka

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
N_POINTS = 4

from collections import namedtuple

class RoadSegment():
	
	Bound = namedtuple('Bound', ['slope', 'offset', 'side'])
	
	def __init__(self, p1:np.array, w1:float, a1:float,
		               p2:np.array, w2:float, a2:float,):
		# p1: road center at start of segment with width w1
		# point p1 has direction angle (radians) a1
		# p2: road center at end of segment with width w2
		# point p2 has direction angle (radians) a2
		self.p1 = p1; self.p2 = p2;
		self.w1 = w1; self.w2 = w2;
		self.a1 = a1; self.a2 = a2;
		# We draw a bounding box around this road segment
		# which has 4 sides, but is not necessarily square
		# we represent it as 4 inequalities
		
		# These are the two bounds on the "cut edge" of the road segment
		self.bounds = [
			self.Bound(slope=np.tan(a1 + np.pi/2), offset=p1, side="<"),
			self.Bound(slope=np.tan(a2 + np.pi/2), offset=p2, side=">"),
		]
		# now we want the two bounds on the road width
		top_1 = np.add(p1, w1*np.array([np.cos(a1 + np.pi/2), np.sin(a1 + np.pi/2)]))
		top_2 = np.add(p2, w2*np.array([np.cos(a2 + np.pi/2), np.sin(a2 + np.pi/2)]))
		upper_bound = self.Bound(slope=(top_2[1]-top_1[1])/(top_2[0]-top_1[0]), offset=top_1, side=">")
		
		b_1 = np.subtract(p1, w1*np.array([np.cos(a1 + np.pi/2), np.sin(a1 + np.pi/2)]))
		b_2 = np.subtract(p2, w2*np.array([np.cos(a2 + np.pi/2), np.sin(a2 + np.pi/2)]))
		lower_bound = self.Bound(slope=(b_2[1]-b_1[1])/(b_2[0]-b_1[0]), offset=b_1, side="<")
		self.bounds.append(upper_bound)
		self.bounds.append(lower_bound)

	@staticmethod
	def check_bound(bound,p):
		print("Bound:\n", bound)
		print("result: ", bound.slope*(p[0]-bound.offset[0]) + bound.offset[1])
		if bound.side == ">" and np.sign(bound.slope) == 1 \
		or bound.side == "<" and np.sign(bound.slope) == -1:
			print("Check ABOVE:")
			res = p[1] >= bound.slope*(p[0]-bound.offset[0]) + bound.offset[1]
			return res
		else:
			print("Check BELOW")
			res =  p[1] < bound.slope*(p[0]-bound.offset[0]) + bound.offset[1]
			return res

	def __contains__(self, p:np.array):
		return all([self.check_bound(b,p) for b in self.bounds])


class Roadrunner():
	def __init__(self, road_center:np.array, road_width:np.array):


		# experiment



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

		# Compute the straight-line distances between points
		self.dists = np.empty(np.shape(road_width))
		for i in range(16):
			self.dists[i] = np.sqrt((road_center[i+1,0]-road_center[i,0])**2 + \
				(road_center[i+1,1]-road_center[i,1])**2)
		self.dists[-1] = self.dists[-2]

		self._dist_to_next = 0.0
		self.p = 0

		self.segments = []

	def new_segment(self, i:int)->callable:
		# Returns a function to represent the next bit of road.
		road_body_frame = self.to_body_frame(road_center[i:i+N_POINTS,:], self.angle[i])

		def fit1(x, a, b, c):

			return a*np.square(x) + b*x + c

		def fit2(x,a,b,c):
			return a/(x+c)
		
		popt_y1, pcov_y1 = curve_fit(fit1, road_body_frame[:,0], road_body_frame[:,1])
		
		fit = fit1; popt_y = popt_y1

		def f(x:np.array or float):

			y = fit(x, *popt_y)
			x_len = 1 if type(x) == float else np.size(x)

			xy = np.hstack([np.reshape(x,(x_len,1)),np.reshape(y, (x_len,1))])
			return self.to_world_frame(xy, rr.angle[rr.p])

		self.segments.append(f)
		if len(self.segments) >= N_POINTS:
			self.segments = self.segments[-N_POINTS-1:]
		return f


	def advance(self, x_new:float, y_new:float)->None:
		# Imagine the road as a set of vectors
		# which are the road center-points with direction pointing to the next point.
		# We want to draw the perpendicular line and
		# determine if we are before that line or after it.
		next_x, next_y = self.road_center[self.p+1]
		
		if np.abs(self.angle[self.p]) < 1e-2: # road is almost straight
			if x_new + self._dist_to_next >= self.dists[self.p]:
				self.p += 1
		else: # road is angled
			# we draw a line perpendicular to the next vector road_dist, road_angle
			# and check which side our point is on.
			slope = np.tan(self.angle[self.p+1] + np.pi/2.0)
			# function is f(x) = slope*(x-next_x) + next_y
			if (slope < 0.0 and slope*(x-next_x)+next_y < 0.0) \
			or (slope > 0.0 and slope*(x-next_x)+next_y > 0.0):
				self.p += 1
				print("advanced p to", self.p)
				self.new_segment(self.p)


	def to_body_frame(self, road_center:np.array, angle:float)->np.array:
		new_center = np.empty(np.shape(road_center))
		new_center[:,0] = np.multiply(road_center[:,0], np.cos(angle)) + \
						  np.multiply(road_center[:,1], np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0], -np.sin(angle)) + \
						  np.multiply(road_center[:,1], np.cos(angle))
		return new_center

	def to_world_frame(self, road_center:np.array, angle:float)->np.array:
		new_center = np.empty(np.shape(road_center))
		new_center[:,0] = np.multiply(road_center[:,0], np.cos(angle)) + \
						  np.multiply(road_center[:,1], -np.sin(angle))
		new_center[:,1] = np.multiply(road_center[:,0], np.sin(angle)) + \
						  np.multiply(road_center[:,1], np.cos(angle))
		return new_center



if __name__ == "__main__":
	# road
	road_center = np.array([
	[2.519, 117.514],
	[10.68, 117.192],
	[22.303, 116.549],
	[30.712, 115.585],
	[40.357, 112.691],
	[50.744, 107.226],
	[50.249, 98.224],
	[48.765, 84.721],
	[47.529, 74.754],
	[47.158, 64.466],
	[47.034, 53.535],
	[47.529, 41.318],
	[48.024, 31.994],
	[48.518, 22.028],
	[58.41, 22.671],
	[68.303, 23.635],
	[77.453, 23.153],
	])

	road_width = np.array([
		6.0,
		6.0,
		5.95,
		5.9,
		5.84,
		5.80,
		5.80,
		5.86,
		5.82,
		5.78,
		5.72,
		5.7,
		5.68,
		5.6,
		5.52,
		5.44,
		5.40,
	])

	rr = Roadrunner(road_center, road_width)
	import matplotlib.pyplot as plt


	print("bound test")

	rs = RoadSegment(road_center[0,:], road_width[0],rr.angle[0],
					 road_center[1,:],road_width[1], rr.angle[1])
	print("ends:", road_center[0,:], road_width[0], rr.angle[0])
	print("ends:", road_center[1,:], road_width[1], rr.angle[1])


	print("Box:\n", rs.bounds)

	mp = 0.5*(road_center[0,:] + road_center[1,:])
	print("midpoint", mp)
	print("midpoint is in: ", mp in rs)
	print("endpoint1 is in: ", road_center[0]+1e-6 in rs)
	print("endpoint2 is out: ", road_center[2] not in rs)

	x = np.linspace(road_center[0,0]-1, road_center[1,0]+1,10)
	for i in range(4):
		plt.plot(x, rs.bounds[i].slope*(x-rs.bounds[i].offset[0]) + rs.bounds[i].offset[1])
	plt.scatter([road_center[0,0], road_center[1,0]],[road_center[0,1], road_center[1,1]],)
	plt.scatter([0.5*(road_center[0,0]+ road_center[1,0]),], [0.5*(road_center[0,1]+ road_center[1,1]),])

	'''
	print("change test")
	coords = rr.to_body_frame(road_center, rr.angle[0])
	print("angle", rr.angle)
	print(coords)
	plt.plot(coords[:,0], coords[:,1],label="T(xy)")
	coords = rr.to_world_frame(coords, rr.angle[0])
	print(coords)
	plt.plot(coords[:,0], coords[:,1], label="T(T(xy))", linewidth=3.0)
	plt.plot(road_center[:,0], road_center[:,1], label="xy", linestyle='--')
	plt.legend()
	print("\n\n")
	'''
	fig2, ax = plt.subplots(1,1)
	dists = np.empty(np.shape(road_width))
	for i in range(16):
		dists[i] = np.sqrt((road_center[i+1,0]-road_center[i,0])**2 + \
			(road_center[i+1,1]-road_center[i,1])**2)
	dists[-1] = dists[-2]
	vx = np.multiply(dists, np.cos(rr.angle))
	vy = np.multiply(dists, np.sin(rr.angle))
	ax.quiver(road_center[:,0], road_center[:,1], vx, vy)

	for i in range(0,len(road_width)-N_POINTS):
		rr.p = i
		f = rr.new_segment(i)
		start = rr.to_body_frame(road_center[i:i+1,:], rr.angle[i])
		end = rr.to_body_frame(road_center[i+N_POINTS:i+N_POINTS+1,:], rr.angle[i])
		
		x = np.linspace(start[0,0], end[0,0], 10)
		
		xfx = rr.to_world_frame(f(x), rr.angle[i])
		xfx = f(x)
		ax.plot(xfx[:,0], xfx[:,1], label="Segment {}".format(i))

	rr.p = 1

	points = np.empty((20,2))
	x = rr.road_center[rr.p,0]
	for i in range(20):

		xy = rr.segments[-2](x)
		print("xy", xy)
		points[i,:] = xy
		x += 1.0*np.cos(rr.angle[rr.p])
		xy[0,1] += 1.*np.sin(rr.angle[rr.p])
		rr.advance(x, xy[0,1])

	plt.scatter(points[:,0], points[:,1], label="test")
	plt.xlim(-50,100)
	plt.ylim(-50,150)
	plt.legend()
	plt.show()