"""

A* grid planning

author: Xudong Zhuang (xzhuangad@connect.ust.hk)
        Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import matplotlib.pyplot as plt
import time 
import numpy as np
import argparse
from numpy.lib.shape_base import _replace_zero_by_x_arrays

show_animation = True # turn it to False to calculate total time consumption


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        param index
        param min_position
        return pos
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

def obstacle_generator(ox, oy, obs_num, obs_length, map_size):
    # generate the lower right point of the square obstacle
    xs = np.random.randint(5, map_size-5, size=obs_num)
    ys = np.random.randint(5, map_size-5, size=obs_num)
    for i in range(obs_num):
        for k in range(obs_length):
            for j in range(obs_length):
                ox.append(xs[i] + k)
                oy.append(ys[i] + j)
    return ox, oy

def calc_total_dist(rx, ry, sx, sy, gx, gy):
    """calculate the total path length"""
    dist = 0
    assert (len(rx) == len(ry))
    if rx[0] != gx and ry[0] != gy:
        rx.insert(0, gx)
        ry.insert(0, gy)
    if rx[len(rx)-1] != sx and ry[len(ry)-1] != sy:
        rx.append(sx)
        ry.append(sy)

    for i in range(len(rx)-1):
        dist += np.linalg.norm([rx[i] - rx[i+1], ry[i] - ry[i+1]])

    return dist


def calc_dist(rx, ry, sx, sy, tx, ty, total_dist):
    dist = 0
    assert (len(rx) == len(ry))
    if rx[len(rx)-1] != sx and ry[len(ry)-1] != sy:
        rx.append(sx)
        ry.append(sy)
    # get the index of target point
    for i in range(len(rx)):
        if rx[i] == tx and ry[i] == ty:
            index = i
    
    for i in range(0, index):
        dist += np.linalg.norm([rx[i] - rx[i+1], ry[i] - ry[i+1]])

    return total_dist - dist

def fit_line(x1, x2, y1, y2):
    if x1 != x2:
        a = (y1 - y2)/(x1 - x2)
        b = y1 - a * x1
        return a, b
    else:
        return x1

def fit_parabolic(x1, x2, x3, y1, y2, y3): 
    if x1 != x2 and x2 != x3:
        A = np.array([[x1**2, x1, 1],
                    [x2**2, x2, 1],
                    [x3**2, x3, 1]])
        B = np.array([y1, y2, y3])
        C = np.linalg.solve(A, B)

        return C[0], C[1], C[2]

def path_smoother(rx, ry, dens):
    rx_new, ry_new = rx.copy(), ry.copy()
    dens = 20
    assert (len(rx_new) == len(ry_new))
    length = len(rx)
    index = 1

    for i in range(1, length-1):

        vec1 = [rx[i-1] - rx[i], ry[i-1] - ry[i]]
        vec2 = [rx[i] - rx[i+1], ry[i] - ry[i+1]]
        cos_theta = ((vec1[0] * vec2[0]) + (vec1[1] * vec2[1]))/(np.linalg.norm(np.array(vec1)) * np.linalg.norm(np.array(vec2)))

        if cos_theta < 1 - 1e-5: # the angle between the pt_i, pt_i-1 and pt_i+1 is not 180

            # for the 1st pattern
            if rx[i-1] != rx[i+1] and rx[i] != rx[i+1] and rx[i] != rx[i-1]:
                a_l, b_l = fit_line(rx[i-1], rx[i+1], ry[i-1], ry[i+1]) # fit the line
                ry_new[index] = (ry[i] + a_l*rx[i]+b_l)/2
                a_p, b_p, c_p = fit_parabolic(rx[i-1], rx[i], rx[i+1], ry[i-1], ry_new[index], ry[i+1]) # fit the parabola

                # get multiple xs between rx_i-1, rx_i+1
                xs = np.linspace(rx[i-1], rx[i+1], dens)
                xs = np.delete(xs, [0, len(xs)-1])

                for j in range(len(xs)):
                    if xs[j] < rx[i]:
                        rx_new.insert(index+j, xs[j])
                        ry_new.insert(index+j, a_p*xs[j]**2 + b_p*xs[j] + c_p)
                    elif xs[j] > rx[i]:
                        rx_new.insert(index+j+1, xs[j])
                        ry_new.insert(index+j+1, a_p*xs[j]**2 + b_p*xs[j] + c_p)
                    else:
                        continue

            # for the 2nd pattern
            else:
                a_l, b_l = fit_line(ry[i-1], ry[i+1], rx[i-1], rx[i+1]) # fit the line
                rx_new[index] = (rx[i] + a_l*ry[i]+b_l)/2
                a_p, b_p, c_p = fit_parabolic(ry[i-1], ry[i], ry[i+1], rx[i-1], rx_new[index], rx[i+1]) # fit the parabola

                ys = np.linspace(ry[i-1], ry[i+1], dens)
                ys = np.delete(ys, [0, len(ys)-1])

                for j in range(len(ys)):
                    if ys[j] < ry[i]:
                        ry_new.insert(index+j, ys[j])
                        rx_new.insert(index+j, a_p*ys[j]**2 + b_p*ys[j] + c_p)
                    elif ys[j] > ry[i]:
                        ry_new.insert(index+j+1, ys[j])
                        rx_new.insert(index+j+1, a_p*ys[j]**2 + b_p*ys[j] + c_p)
                    else:
                        continue

            # update index
            index = index + dens - 1
                
        else:
            index += 1

    return rx_new, ry_new

def interpolate(rx, ry):
    rx_new, ry_new = rx.copy(), ry.copy()
    length = len(rx_new)
    index = 0
    for i in range(length-1):
        rx_new.insert(index+1, (rx_new[index] + rx_new[index+1])/2)
        ry_new.insert(index+1, (ry_new[index] + ry_new[index+1])/2)
        index = index+2
    return rx_new, ry_new

def main():
    print(__file__ + " start!!")

    parser = argparse.ArgumentParser()
    parser.add_argument('-spt', '--starting_point', 
                        help='starting point',
                        dest='spt',
                        default= [2.0, 2.0],
                        type=list)

    parser.add_argument('-tpt', '--target_point', 
                        help='target point',
                        dest = 'tpt',
                        default= [48.0, 48.0],
                        type=list)

    parser.add_argument('-ms', '--map_size', 
                        help='side length of square map',
                        dest = 'ms',
                        default= 50,
                        type=int)

    parser.add_argument('-ol', '--obstacle_length', 
                        help='side length of square map',
                        dest= 'ol',
                        default= 10,
                        type=int)

    parser.add_argument('-ptl', '--point_list', 
                        help='a list of point representing the left lower points of obstacles',
                        dest = 'ptl',
                        default= [[15, 15], [35, 30], [3, 35], [23, 3]],
                        type=list)

    parser.add_argument('-d', '--dense', 
                        help='dense of path smoother',
                        dest = 'dense',
                        default= 20,
                        type=int)

    args = parser.parse_args()

    # start and goal position
    sx = args.spt[0]  # [m]
    sy = args.spt[1]  # [m]
    gx = args.tpt[0]  # [m]
    gy = args.tpt[1]   # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1  # [m]
    obs_length = args.ol
    map_size = args.ms
    left_lower_pts_list = args.ptl
    dens = args.dense

    ox, oy = [], []

    # set the boundary of the map
    for i in range(0, map_size):
        ox.append(i)
        oy.append(0)
    for i in range(0, map_size):
        ox.append(map_size)
        oy.append(i)
    for i in range(0, map_size+1):
        ox.append(i)
        oy.append(map_size)
    for i in range(0, map_size+1):
        ox.append(0)
        oy.append(i)

        # set the obstables inside the map with different obs_length but fixed left lower point
        for k in range(len(left_lower_pts_list)):
            for i in range(0, obs_length):
                for j in range(0, obs_length):
                    ox.append(left_lower_pts_list[k][0] + i)
                    oy.append(left_lower_pts_list[k][1] + j)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
        #plt.show() # for showing map only


    start_time = time.time()

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    # get the original planned path
    rx, ry = a_star.planning(sx, sy, gx, gy) 
    # get the interpolated version of the path
    rx_interp, ry_interp = interpolate(rx, ry)
    # get the smoothed version of the path
    rx_smoothed, ry_smoothed = path_smoother(rx_interp, ry_interp, dens)
    end_time = time.time()
    print("The total time for A* path planner is " + str(float(end_time - start_time)))

    total_path_length = calc_total_dist(rx, ry, sx, sy, gx, gy)
    print("The distance of the A* planned path is " + str(total_path_length))

    if show_animation:  # pragma: no cover
        plt.plot(rx_interp, ry_interp, ".k")
        plt.plot(rx, ry, ".r")
        plt.plot(rx, ry, "-r", label = "original path")
        plt.plot(rx_smoothed, ry_smoothed, "-b", label = "smoothed path")
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.legend(fontsize = 20)
        plt.pause(0.001)
        plt.show()



if __name__ == '__main__':
    main()
