"""

D* grid planning

author: Xudong Zhuang (xzhuangad@connect.ust.hk)
        Nirnay Roy

See Wikipedia article (https://en.wikipedia.org/wiki/D*)

"""
import math
from sys import maxsize
import matplotlib.pyplot as plt
import time
import numpy as np

show_animation = True # turn it to False to calculate total time consumption of D* algorithm


class State:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) + math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:

    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue

            self.map[x][y].set_state("#")


class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()

    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()
        self.remove(x)

        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
        elif k_old == x.h:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        if y.parent != x and x.h > y.h + x.cost(y) \
                                and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()

    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)

    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):

        rx = []
        ry = []

        self.open_list.add(end)

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")
        s = start
        s = s.parent
        s.set_state("e")
        tmp = start

        while tmp != end:
            tmp.set_state("*")
            rx.append(tmp.x)
            ry.append(tmp.y)
            if show_animation:
                plt.plot(rx, ry, "-r")
                plt.pause(0.01)
            if tmp.parent.state == "#":
                self.modify(tmp)
                continue
            tmp = tmp.parent
        tmp.set_state("e")

        return rx, ry

    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break


def calc_total_dist(rx, ry):
    """calculate the total path length"""
    dist = 0
    for i in range(len(rx)-1):
        dist += np.linalg.norm([rx[i] - rx[i+1], ry[i] - ry[i+1]])

    return dist

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
    obs_length = 10
    map_size = 50
    m = Map(100, 100)
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
    # set the obstable inside the map
    for i in range(0, obs_length):
        for j in range(0, obs_length):
            ox.append(15 + i)
            oy.append(15 + j)
    
    for i in range(0, obs_length):
        for j in range(0, obs_length):
            ox.append(35 + i)
            oy.append(30 + j)

    for i in range(0, obs_length):
        for j in range(0, obs_length):
            ox.append(3 + i)
            oy.append(35 + j)

    for i in range(0, obs_length):
        for j in range(0, obs_length):
            ox.append(23 + i)
            oy.append(3 + j)

    #print([(i, j) for i, j in zip(ox, oy)])
    m.set_obstacle([(i, j) for i, j in zip(ox, oy)])

    sx = 2
    sy = 2
    gx = 48
    gy = 48

    start = [sx, sy]
    goal = [gx, gy]
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start[0], start[1], "og")
        plt.plot(goal[0], goal[1], "xb")
        plt.axis("equal")

    start = m.map[start[0]][start[1]]
    end = m.map[goal[0]][goal[1]]

    # run the algorithm and get the time
    start_time = time.time()
    dstar = Dstar(m)
    rx, ry = dstar.run(start, end)
    # get the interpolated version of the path
    rx_interp, ry_interp = interpolate(rx, ry)
    # get the smoothed version of the path
    rx_smoothed, ry_smoothed = path_smoother(rx_interp, ry_interp, 20)
    end_time = time.time()
    print("The total time for D* path planner is " + str(float(end_time - start_time)))


    total_path_length = calc_total_dist(rx, ry)
    print("The distance of the planned path is " + str(total_path_length))
    total_path_length_smoothed = calc_total_dist(rx_smoothed, ry_smoothed)
    print("The distance of the smoothed path is " + str(total_path_length_smoothed))

    if show_animation:
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
