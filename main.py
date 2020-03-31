"""
Coverage path planning (CPP) algorithm implementation for a mobile robot
equipped with 4 ranger sensors (front, back, left and right)
for obstacles detection.

author: Ruslan Agishev (agishev_ruslan@mail.ru)
modified by : Minkyu Kim (steveminq@utexas.edu)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle, Circle
import numpy as np
import math
from grid_map import GridMap
from grid_based_sweep_coverage_path_planner import planning
import time
from tools import define_polygon, polygon_contains_point

fig,axes=plt.subplots(nrows=2,ncols=1,figsize=(10,20))

def plot_grid_map(gridmap, ax=None):
        # print("plot_grid_map")
        all_axes=plt.gcf().get_axes()
        ax=all_axes[1]
        grid_data = np.reshape(np.array(self.data), (int(self.height), int(self.width)))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(grid_data, cmap="Blues", vmin=0.0, vmax=1.0)
        # plt.axis("equal")
        return heat_map



def plot_robot(pose, params):
        print("robot")
	r = params.sensor_range_m
        # plt.axis("equal")
        # ax = plt.gca()
        axes[1].plot([pose[0]-r*np.cos(pose[2]), pose[0]+r*np.cos(pose[2])],
                [pose[1]-r*np.sin(pose[2]), pose[1]+r*np.sin(pose[2])], '--', color='b')
        axes[1].plot([pose[0]-r*np.cos(pose[2]+np.pi/2), pose[0]+r*np.cos(pose[2]+np.pi/2)],
                [pose[1]-r*np.sin(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)], '--', color='b')
        
	# plt.plot([pose[0]-r*np.cos(pose[2]), pose[0]+r*np.cos(pose[2])],
                # [pose[1]-r*np.sin(pose[2]), pose[1]+r*np.sin(pose[2])], '--', color='b')
	# plt.plot([pose[0]-r*np.cos(pose[2]+np.pi/2), pose[0]+r*np.cos(pose[2]+np.pi/2)],
                # [pose[1]-r*np.sin(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)], '--', color='b')
        # plt.plot(pose[0], pose[1], 'ro', markersize=5)
        circle= Circle((pose[0], pose[1]),r,linewidth=1,edgecolor='k',facecolor='k',alpha=0.4 )
        axes[1].add_patch(circle)
        # plt.plot(pose[0], pose[1], 'ro', markersize=40, alpha=0.1)
        # print("plot_circle")
        # ax.arrow(pose[0], pose[1], 0.05 * np.cos(pose[2]), 0.05 * np.sin(pose[2]),
              # head_length=0.1, head_width=0.1)
        axes[1].arrow(pose[0], pose[1], 0.05 * np.cos(pose[2]), 0.05 * np.sin(pose[2]),
              head_length=0.1, head_width=0.1)


def obstacle_check(pose, gridmap, params):
	gmap = gridmap
	r = int(100*params.sensor_range_m)
	back = [pose[0]-r*np.cos(pose[2]), pose[1]-r*np.sin(pose[2])]
	front = [pose[0]+r*np.cos(pose[2]), pose[1]+r*np.sin(pose[2])]
	right = [pose[0]+r*np.cos(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)]
	left = [pose[0]-r*np.cos(pose[2]+np.pi/2), pose[1]-r*np.sin(pose[2]+np.pi/2)]

	pi = np.array(pose[:2], dtype=int)
	backi = np.array(back, dtype=int)
	fronti = np.array(front, dtype=int)
	lefti = np.array(left, dtype=int)
	righti = np.array(right, dtype=int)

	obstacle = {
		'front': 0,
		'back':  0,
		'right': 0,
		'left':  0,
                }

	for i in np.arange(min(pi[0], fronti[0]), max(pi[0], fronti[0])+1):
		for j in np.arange(min(pi[1], fronti[1]), max(pi[1], fronti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('FRONT collision')
				obstacle['front'] = 1

	for i in np.arange(min(pi[0], backi[0]), max(pi[0], backi[0])+1):
		for j in np.arange(min(pi[1], backi[1]), max(pi[1], backi[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('BACK collision')
				obstacle['back'] = 1

	for i in np.arange(min(pi[0], lefti[0]), max(pi[0], lefti[0])+1):
		for j in np.arange(min(pi[1], lefti[1]), max(pi[1], lefti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('LEFT collision')
				obstacle['left'] = 1

	for i in np.arange(min(pi[0], righti[0]), max(pi[0], righti[0])+1):
		for j in np.arange(min(pi[1], righti[1]), max(pi[1], righti[1])+1):
			m = min(j, gmap.shape[0]-1); n = min(i, gmap.shape[1]-1)
			if gmap[m,n]:
				# print('RIGHT collision')
				obstacle['right'] = 1

	return obstacle



def left_shift(pose, r):
	left = [pose[0]+r*np.cos(pose[2]+np.pi/2), pose[1]+r*np.sin(pose[2]+np.pi/2)]
	return left
def right_shift(pose, r):
	right = [pose[0]-r*np.cos(pose[2]+np.pi/2), pose[1]-r*np.sin(pose[2]+np.pi/2)]
	return right
def back_shift(pose, r):
	back = pose
	back[:2] = [pose[0]-r*np.cos(pose[2]), pose[1]-r*np.sin(pose[2])]
	return back
def forward_shift(pose, r):
	forward = pose
	forward[:2] = [pose[0]+r*np.cos(pose[2]), pose[1]+r*np.sin(pose[2])]
	return forward
def turn_left(pose, yaw=np.pi/2*np.random.uniform(0.2, 0.6)):
	pose[2] -= yaw
	return pose
def turn_right(pose, yaw=np.pi/2*np.random.uniform(0.2, 0.6)):
	pose[2] += yaw
	return pose
def slow_down(state, params, dv=0.1):
	if state[3]>params.min_vel:
		state[3] -= dv
	return state

def visualize(traj, pose, params):
        # axes = plt.gca()
        # plt.plot(traj[:,0], traj[:,1], 'g')
        # plot_robot(pose, params)
        # plt.legend()
	axes[1].set_xlim([-2.5, 2.5])
	axes[1].set_ylim([-2.5, 2.5])
        axes[1].plot(traj[:,0], traj[:,1], 'g')
        # plt.axis("equal")
        plot_robot(pose, params)
        axes[1].legend()

def visualize_coverage(poseset):
    # ax = plt.gca()
    for pose in poseset:
        circle= Circle((pose[0], pose[1]),0.25,linewidth=1,edgecolor='r',facecolor='r',alpha=0.15 )
        # ax.add_patch(rect)
        # ax.add_patch(circle)
        axes[1].add_patch(circle)
        # plt.plot(pose[0], pose[1], 'ro', markersize=30, alpha=0.10)
def update_coveragemap(pose,coveragemap):
    coveragemap.set_value_from_xy_pos(pose[0],pose[1],5.0)

def draw_obstacles_gridmap(gridmap,obstacles):
    ax = plt.gca()
    w = gridmap.map_width_m; l = gridmap.map_length_m
    # ax.set_xlim([-w/2, w/2])
    # ax.set_ylim([-l/2, l/2])
    boundaries = gridmap.flight_area_vertices
    ax.add_patch( Polygon(boundaries, linewidth=2, edgecolor='k',facecolor='none') )
    for k in range(len(obstacles)):
        ax.add_patch( Polygon(obstacles[k]+gridmap.map_center, color='k', zorder=10) )



def motion(state, goal, params):
	# state = [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
	dx = goal[0] - state[0]
	dy = goal[1] - state[1]
	goal_yaw = math.atan2(dy, dx)
	K_theta = 3.2
	state[4] = K_theta*math.sin(goal_yaw - state[2]) # omega(rad/s)
	state[2] += params.dt*state[4] # yaw(rad)

	dist_to_goal = np.linalg.norm(goal - state[:2])
	K_v = 0.1
	state[3] += K_v*dist_to_goal
	if state[3] >= params.max_vel: state[3] = params.max_vel
	if state[3] <= params.min_vel: state[3] = params.min_vel

	dv = params.dt*state[3]
	state[0] += dv*np.cos(state[2]) # x(m)
	state[1] += dv*np.sin(state[2]) # y(m)

	return state

def collision_avoidance(state, gridmap, params):
	pose_grid = gridmap.meters2grid(state[:2])
	boundary = obstacle_check([pose_grid[0], pose_grid[1], state[2]], gridmap.gmap, params)
	# print(boundary)

	if boundary['right'] or boundary['front']:
		# state = back_shift(state, 0.03)
		state = slow_down(state, params)
		state = turn_left(state, np.radians(40))
		# state = forward_shift(state, 0.02)
	elif boundary['left']:
		# state = back_shift(state, 0.03)
		state = slow_down(state, params)
		state = turn_right(state, np.radians(40))
		# state = forward_shift(state, 0.02)
	return state

def define_flight_area(initial_pose):
	plt.grid()
	while True:
		try:
			num_pts = int( input('Enter number of polygonal vertixes: ') )
			break
		except:
			print('\nPlease, enter an integer number.')
	while True:
		flight_area_vertices = define_polygon(num_pts)
		if polygon_contains_point(initial_pose, flight_area_vertices):
			break
		plt.clf()
		plt.grid()
		print('The robot is not inside the flight area. Define again.')
	return flight_area_vertices

class Params:
	def __init__(self):
		self.numiters = 1000
		self.animate = 1
		self.dt = 0.1
		self.goal_tol = 0.25
		self.max_vel = 0.5 # m/s
		self.min_vel = 0.1 # m/s
		self.sensor_range_m = 0.25 # m
		self.time_to_switch_goal = 5.0 # sec #inactive for now
		self.sweep_resolution = 0.4 # m

def main():
	obstacles = [
		# np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]) + np.array([-1.0, 0.5]),
		# np.array([[0.7, -0.9], [1.3, -0.9], [1.3, -0.8], [0.7, -0.8]]) + np.array([-1.0, 1.0]),
		# np.array([[0.7, -0.9], [0.8, -0.9], [0.8, -0.3], [0.7, -0.3]]) + np.array([-1.5, 1.0]),        
	
		np.array([[-0.3, -0.4], [0.3, -0.4], [0.3, 0.1], [-0.3, 0.1]]) * 0.5
	]
	# initial state = [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
	state = np.array([0, 0.2, np.pi/2, 0.0, 0.0])
	traj = state[:2]
	params = Params()
	# plt.figure(figsize=(10,10))

	flight_area_vertices = define_flight_area(state[:2])
        posset=[]
	# flight_area_vertices = np.array([[-1, -1], [-0.3, -1], [-0.3, -0.4], [0.3, -0.4], [0.3, -1], [1,-1], [1,1], [-1,1]])
	gridmap = GridMap(flight_area_vertices, state[:2])
        gridmap.add_obstacles_to_grid_map(obstacles)

        #obstacle x, y coordinates
        ox = flight_area_vertices[:,0].tolist() + [flight_area_vertices[0,0]]
        oy = flight_area_vertices[:,1].tolist() + [flight_area_vertices[0,1]]
        reso = params.sweep_resolution
        goal_x, goal_y, gmap, covmap = planning(ox, oy, reso)
        # covmap.plot_grid_map(axes[0])

	# goal = [x, y], m
	goali = 0
	goal = [goal_x[goali], goal_y[goali]]
	t_prev_goal = time.time()
	gridmap.draw_map(obstacles)
        iter=0

	# while True:
	for _ in range(params.numiters):
		state = motion(state, goal, params)
		state = collision_avoidance(state, gridmap, params)

                posset.append([state[0],state[1]])
                update_coveragemap(state,covmap)

		goal_dist = np.linalg.norm(goal - state[:2])
		# print('Distance to goal %.2f [m]:' %goal_dist)
		t_current = time.time()
		# if goal_dist < params.goal_tol or (t_current - t_prev_goal) > params.time_to_switch_goal: # goal is reached
		if goal_dist < params.goal_tol: # goal is reached
                    print('Switching to the next goal.')
                    print('Time from the previous reached goal:', t_current - t_prev_goal)
                    if goali < len(goal_x) - 1:
                        goali += 1
                    else:
                        break
                    t_prev_goal = time.time()
                    goal = [goal_x[goali], goal_y[goali]]


		traj = np.vstack([traj, state[:2]])
		
		if params.animate:
                        axes[1].cla()
                        # plt.cla()
                        gridmap.draw_map(obstacles,axes[1]) #mk
                        axes[1].plot(goal_x, goal_y)
                        axes[1].plot(goal[0], goal[1], 'ro', markersize=12, label='Goal position', zorder=20)
                        visualize(traj, state, params)
                        visualize_coverage(posset)
                        # plt.plot(goal_x, goal_y)
                        # plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal position', zorder=20)
                        # visualize(traj, state, params)
                        # visualize_coverage(posset)
                        covmap.plot_grid_map(axes[0])
                        plt.pause(0.01)
                iter=iter+1
                if iter==1:
                    plt.savefig('planned_coverage_path.png',dpi=300)
                    # covmap2.plot_grid_map(axes[0])

	print('Mission is complete!')
	plt.plot(goal_x, goal_y)
	visualize(traj, state, params)
	plt.show()
        # input()

if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
            pass
		
