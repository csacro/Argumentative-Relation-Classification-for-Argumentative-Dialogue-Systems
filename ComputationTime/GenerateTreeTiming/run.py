import util
import time
import logging

from generate_tree import tmg
from generate_tree import bilp
import Stats

import numpy as np
from datetime import datetime


def main():
    for i in range(len(Stats.n)):
        for c in range(Stats.count[i]):
            print("i: " + str(i) + ", c: " + str(c))

            max_circle_matrix = create_max_circle_matrix(Stats.n[i])

            timer = datetime.now()
            result_max_circle_1 = tmg.make_relation_matrix(np.copy(max_circle_matrix))
            Stats.max_circle_tmg_time[i] += datetime.now() - timer
            print(str(datetime.now()) + " max 1 " + str(Stats.max_circle_tmg_time[i]))

            #timer = datetime.now()
            #result_max_circle_2 = bilp.make_relation_matrix(np.copy(max_circle_matrix))
            #Stats.max_circle_bilp_time[i] += datetime.now() - timer
            #print(str(datetime.now()) + " max 2 " + str(Stats.max_circle_bilp_time[i]))

            no_circle_matrix = create_no_circle_matrix(result_max_circle_1)

            timer = datetime.now()
            result_no_circle_1 = tmg.make_relation_matrix(np.copy(no_circle_matrix))
            Stats.no_circle_tmg_time[i] += datetime.now() - timer
            print(str(datetime.now()) + " no 1 " + str(Stats.no_circle_tmg_time[i]))

            #timer = datetime.now()
            #result_no_circle_2 = bilp.make_relation_matrix(np.copy(no_circle_matrix))
            #Stats.no_circle_bilp_time[i] += datetime.now() - timer
            #print(str(datetime.now()) + " no 2 " + str(Stats.no_circle_bilp_time[i]))

            single_circle_matrix = create_single_circle_matrix(Stats.n[i])

            timer = datetime.now()
            result_single_circle_1 = tmg.make_relation_matrix(np.copy(single_circle_matrix))
            Stats.single_circle_tmg_time[i] += datetime.now() - timer
            print(str(datetime.now()) + " single 1 " + str(Stats.single_circle_tmg_time[i]))

            #timer = datetime.now()
            #result_single_circle_2 = bilp.make_relation_matrix(np.copy(single_circle_matrix))
            #Stats.single_circle_bilp_time[i] += datetime.now() - timer
            #print(str(datetime.now()) + " single 2 " + str(Stats.single_circle_bilp_time[i]))

    util.init_logging(logging.DEBUG, "GenerateTreeTiming_"  + time.strftime("%Y%m%d-%H%M%S"))
    Stats.log_results()
    exit(0)


def create_single_circle_matrix(size):
    mat = np.random.rand(size+1, size+1)
    np.fill_diagonal(mat[:, 1:], 1)
    mat[0][1] = np.random.rand(1)
    mat[size][1] = 1
    return mat


def create_max_circle_matrix(size):
    mat = np.random.rand(size+1, size+1)
    np.fill_diagonal(mat, 1)
    return mat


def create_no_circle_matrix(m):
    repl = m==0
    s = np.count_nonzero(repl)
    rand_vals = np.random.rand(s)
    m[repl] = rand_vals
    return m


if __name__ == "__main__":
    main()
