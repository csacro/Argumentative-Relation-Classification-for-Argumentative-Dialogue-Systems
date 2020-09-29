import logging
import numpy as np


def make_relation_matrix(prob_matrix):
    # get most probable relation and create relation_matrix
    relation_row = np.argmax(prob_matrix, axis=0)
    relation_matrix = __relation_row_to_matrix(relation_row)
    # find out which nodes are not in the tree with the Major Claim as root
    _, unknown_nodes = __get_node_status(relation_matrix)
    # find/detect and procress circles
    while unknown_nodes:
        circle = []
        logging.debug("calling find circle")
        __find_circle(relation_matrix, unknown_nodes, unknown_nodes[0], circle)
        if circle:
            logging.debug("calling process circle")
            __process_circle(circle, prob_matrix, relation_matrix)

    return relation_matrix


def __find_circle(relation_matrix, unknown_nodes, start: int, circle):
    logging.debug("start: " + str(start))
    if not circle:
        circle.append(start)
        cur_node = -1
        children = __get_list_of_children(relation_matrix, start)
    else:
        cur_node = circle[len(circle) - 1]
        unknown_nodes.remove(cur_node)
        children = __get_list_of_children(relation_matrix, cur_node)

    logging.debug("circle: " + str(circle))
    children = [i for i in children if i in unknown_nodes]
    logging.debug("children: " + str(children))

    while circle:
        if not children and circle == [start]:
            # no circle was found (start node is not part of circle)
            unknown_nodes.remove(start)
            circle.clear()
            logging.debug("no circle found" + str(circle))
            return True
        if not children and cur_node != start:
            # branch detected (return to node in circle)
            circle.pop(len(circle) - 1)
            logging.debug("branch detected: " + str(circle))
            return False
        if start in children:
            # circle found (start node is part of circle)
            unknown_nodes.remove(start)
            logging.debug("circle found: " + str(circle))
            return True
        while children:
            circle.append(children.pop(0))
            if __find_circle(relation_matrix, unknown_nodes, start, circle):
                logging.debug("terminate -> circle is: " + str(circle))
                return True
    logging.critical("out of loop: this should not happen !")
    return False


def __process_circle(circle, prob_matrix, relation_matrix):
    assert (prob_matrix.shape[0] == prob_matrix.shape[1])
    node_to_child = circle[0]
    node_to_parent = 0
    max_prob = prob_matrix[node_to_child][node_to_parent]
    circle_nodes = __get_list_of_circle_nodes(circle, relation_matrix)

    for c in circle:
        for p in range(0, prob_matrix.shape[0]):
            if p != c and p not in circle_nodes and prob_matrix[p][c] > max_prob:
                max_prob = prob_matrix[p][c]
                node_to_child = c
                node_to_parent = p
    logging.debug(str(node_to_child) + " --> " + str(node_to_parent))
    relation_matrix[:, node_to_child] = [0] * len(relation_matrix[:, node_to_child])
    relation_matrix[node_to_parent][node_to_child] = 1


def __get_list_of_circle_nodes(circle, relation_matrix):
    circle_nodes = []
    for c in circle:
        circle_nodes.append(c)
        add_list = [i for i in __get_list_of_children(relation_matrix, c) if i not in circle]
        while add_list:
            cur_node = add_list.pop(0)
            circle_nodes.append(cur_node)
            children = [i for i in __get_list_of_children(relation_matrix, cur_node) if i not in circle]
            add_list.extend(children)
    return circle_nodes


def __get_node_status(relation_matrix, root: int = 0):
    node_list = __traverse_tree(relation_matrix, root)
    known_nodes = [i for i in range(0, len(node_list)) if node_list[i]]
    unknown_nodes = [i for i in range(0, len(node_list)) if not node_list[i]]
    return known_nodes, unknown_nodes


def __traverse_tree(relation_matrix, root: int = 0):
    assert (relation_matrix.shape[0] == relation_matrix.shape[1])
    visited_nodes = [False] * relation_matrix.shape[0]
    # handle root (might point to itself)
    visited_nodes[root] = True
    add_list = __get_list_of_children(relation_matrix, root)
    try:
        add_list.remove(root)
    except ValueError:
		# root is not connected to itself
        pass
    # handle rest
    while add_list:
        cur_node = add_list.pop(0)
        visited_nodes[cur_node] = True
        children = __get_list_of_children(relation_matrix, cur_node)
        add_list.extend(children)
    return visited_nodes


def __get_list_of_children(relation_matrix, parent: int):
    row = relation_matrix[parent]
    children = []
    for i in range(0, len(row)):
        if row[i] != 0:
            children.append(i)
    return children


def __relation_row_to_matrix(relation_row):
    l = len(relation_row)
    relation_matrix = np.zeros((l, l))
    for i in range(0, l):
        relation_matrix[int(relation_row[i])][i] = 1
    return relation_matrix
