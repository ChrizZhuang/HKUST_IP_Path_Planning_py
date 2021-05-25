"""

A* grid planning with polynomial smoother

author: Xudong Zhuang (xzhuangad@connect.ust.hk)
        Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import matplotlib.pyplot as plt
import time
import numpy as np
import sympy as sym

show_animation = True


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

        :param index:
        :param min_position:
        :return:
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

def select_index(rx, ry, num_sec):
    """select index with respect to number of segmentation"""
    assert (len(rx) == len(ry))
    num_pt = num_sec + 1
    num_interval = num_pt - 1
    index_list = []
    length_interval = len(rx) // num_interval
    for i in range(num_pt):
        index_list.append(i*length_interval)
    if index_list[len(index_list)-1] != len(rx)-1:
        index_list[len(index_list)-1] = len(rx)-1
    return index_list

def get_time(curr_dist, total_path_length, ts, tf):
    return curr_dist * (tf - ts)/total_path_length

def get_data(rx, ry, times, selected_index):
    rx_selected = []
    ry_selected = []
    time_selected = []
    for i in range(len(selected_index)):
        rx_selected.append(rx[selected_index[i]])
        ry_selected.append(ry[selected_index[i]])
        time_selected.append(times[selected_index[i]])

    return rx_selected, ry_selected, time_selected

def calc_parameter_single_sec(a_init, v_init, p_init, p_final, t_start, t_final):
    A = np.array([[6 * t_start, 2, 0, 0],
             [3 * t_start**2, 2 * t_start, 1, 0],
             [t_start**3, t_start**2, t_start, 1],
             [t_final**3, t_final**2, t_final, 1]])
    B = np.array([a_init, v_init, p_init, p_final])
    return np.linalg.inv(A).dot(B)

def calc_parameter_last_sec(a_init, a_final, v_init, v_final, p_init, p_final, t_start, t_final):
    A = np.array([[20 * t_start**3, 12 * t_start**2, 6 * t_start, 2, 0, 0],
             [20 * t_final**3, 12 * t_final**2, 6 * t_final, 2, 0, 0],
             [5 * t_start**4, 4 * t_start**3, 3 * t_start**2, 2 * t_start, 1, 0],
             [5 * t_final**4, 4 * t_final**3, 3 * t_final**2, 2 * t_final, 1, 0],
             [t_start**5, t_start**4, t_start**3, t_start**2, t_start, 1],
             [t_final**5, t_final**4, t_final**3, t_final**2, t_final, 1]])
    B = np.array([a_init, a_final, v_init, v_final, p_init, p_final])
    return np.linalg.inv(A).dot(B)

def calc_spline_single(num_sec, pos_selected, time_list):
    """ calculate parameters for x or y """
    a_s, b_s, c_s, d_s = [], [], [], []
    accel_list = [0]
    vel_list = [0]
    # calculate the parameters for x except the last segment
    for i in range(1, num_sec):
        para_list = calc_parameter_single_sec(accel_list[i-1], vel_list[i-1], pos_selected[i-1], 
                    pos_selected[i], time_list[i-1], time_list[i])
        a_s.append(para_list[0])
        b_s.append(para_list[1])
        c_s.append(para_list[2])
        d_s.append(para_list[3])
        accel_list.append(6 * para_list[0] * time_list[i] + 2 * para_list[1])
        vel_list.append(3 * para_list[0] * time_list[i]**2 + 2 * para_list[1] * time_list[i] + para_list[2])
    # calculate the parameters for x for the last segment
    para_list = calc_parameter_last_sec(accel_list[len(accel_list)-1], 0, vel_list[len(vel_list)-1], 0, 
                pos_selected[len(pos_selected)-2], pos_selected[len(pos_selected)-1], time_list[len(time_list)-2], time_list[len(time_list)-1])
    a_s.append(para_list[0])
    b_s.append(para_list[1])
    c_s.append(para_list[2])
    d_s.append(para_list[3])
    e = para_list[4]
    f = para_list[5]

    return a_s, b_s, c_s, d_s, e, f

def calc_spline(num_sec, rx_selected, ry_selected, time_list):
    """calculate splines for x and y"""
    a_xs, b_xs, c_xs, d_xs, e_x, f_x = calc_spline_single(num_sec, rx_selected, time_list)
    a_ys, b_ys, c_ys, d_ys, e_y, f_y = calc_spline_single(num_sec, ry_selected, time_list)
    return [[a_xs, b_xs, c_xs, d_xs, e_x, f_x], [a_ys, b_ys, c_ys, d_ys, e_y, f_y]]


def get_positions(a_s, b_s, c_s, d_s, e, f, time_list, time, num_sec):
    # get the index of spline
    for i in range(num_sec):
        if time >= time_list[i] and time <= time_list[i+1]:
            index = i
    
    # return the position
    if index == num_sec: 
        return a_s[index] * time**5 + b_s[index] * time**4 + c_s[index] * time**3 + d_s[index] * time**2 + e * time + f
    else:
        return a_s[index] * time**3 + b_s[index] * time**2 + c_s[index] * time + d_s[index]





def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 45.0  # [m]
    gy = 45.0  # [m]
    grid_size = 2.0  # [m]
    robot_radius = 1.0  # [m]
    t_start = 0 # [s]
    t_final = 2 # [s]


    # set obstacle positions
    ox, oy = [], []
    # set the boundary of the map
    for i in range(0, 50):
        ox.append(i)
        oy.append(0)
    for i in range(0, 50):
        ox.append(50.0)
        oy.append(i)
    for i in range(0, 50):
        ox.append(i)
        oy.append(50.0)
    for i in range(0, 50):
        ox.append(0)
        oy.append(i)
    # set the obstable inside the map
    for i in range(0, 20):
        ox.append(20)
        oy.append(i)
    for i in range(0, 30):
        ox.append(30)
        oy.append(50 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    start_time = time.time()
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    end_time = time.time()
    print("The total time for A* path planner is " + str(float(end_time - start_time)))

    #rx = np.linspace(sx, gx, 100).tolist()
    #rx.reverse()
    #ry = np.linspace(sy, gy, 100).tolist()
    #ry.reverse()
    total_path_length = calc_total_dist(rx, ry, sx, sy, gx, gy)
    assert (len(rx) == len(ry))
    print("The distance of the planned path is " + str(total_path_length))

    dist_list = []
    time_list = []
    for i in range(len(rx)):
        curr_dist = calc_dist(rx, ry, sx, sy, rx[i], ry[i], total_path_length)
        dist_list.append(curr_dist)
        time_list.append(get_time(curr_dist, total_path_length, t_start, t_final))
    assert(len(dist_list) == len(rx))
    
    num_sec = 20
    # calculate the splines for x and y with respect to time
    index_list = select_index(rx, ry, num_sec)

    rx_selected, ry_selected, time_selected = get_data(rx, ry, time_list, index_list)
    #print(time_selected)
    # rx_selected, ry_selected, time_selected are all from goal to starting point
    rx_selected.reverse()
    ry_selected.reverse()
    time_selected.reverse()
    [[a_xs, b_xs, c_xs, d_xs, e_x, f_x], [a_ys, b_ys, c_ys, d_ys, e_y, f_y]] = calc_spline(num_sec, rx_selected, ry_selected, time_selected)
    print(time_selected)
    print(rx_selected)
    print(ry_selected)
    print(str(a_xs[1]) + " " + str(a_ys[1]))
    print(str(b_xs[1]) + " " + str(b_ys[1]))
    print(str(c_xs[1]) + " " + str(c_ys[1]))
    print(str(d_xs[1]) + " " + str(d_ys[1]))
    #print(e_x)
    #print(f_x)
    #print(a_xs)
    times = np.linspace(0, 0.5, 50)
    xs = []
    ys = []
    #x = get_positions(a_xs, b_xs, c_xs, d_xs, e_x, f_x, time_selected, 0.01, num_sec)
    #y = get_positions(a_ys, b_ys, c_ys, d_ys, e_y, f_y, time_selected, 0.01, num_sec)
    #print(str(x) + " " + str(y))
    for t in times:
        xs.append(get_positions(a_xs, b_xs, c_xs, d_xs, e_x, f_x, time_selected, t, num_sec))
        ys.append(get_positions(a_ys, b_ys, c_ys, d_ys, e_y, f_y, time_selected, t, num_sec))
        




        # see if it is overlapped by obstacles


    
    if show_animation:  # pragma: no cover
        plt.xlim(-10, 70)
        plt.ylim(-10, 70)
        plt.plot(rx, ry, "-r")
 
        plt.plot(xs, ys, "ro")
        plt.plot(rx_selected, ry_selected, "go")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
