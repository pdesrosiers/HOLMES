import networkx as nx
import gudhi
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
import csv
from tqdm import tqdm
import pandas as pd
import os
"""
Author: Xavier Roy-Pomerleau <xavier.roy-pomerleau.1@ulaval.ca>
In this module we compute the Betti numbers of a dataset and its randomized instances. We also plot the Betti number
distribution and the graph representing the real data.
"""

def facet_list_to_graph(facet_list):
    """Convert a facet list to a bipartite graph"""
    g = nx.Graph()
    for f, facet in enumerate(facet_list):
        for v in facet:
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g

def facet_list_to_bipartite(facet_list):
    """
    Convert a facet list to a bipartite graph. It also returns one of the set of the bipartite graph. This is usefull
    in the case where the graph is disconnected in the sense that no path exists between one or more node(s) and all
    the others. This case raises this exception when we want to plot the graph with networkX
    AmbiguousSolution : Exception – Raised if the input bipartite graph is disconnected and no container with all nodes
    in one bipartite set is provided. When determining the nodes in each bipartite set more than one valid solution is
    possible if the input graph is disconnected.
    In this case, we can use TODO to specify one of the node set (here the facets).
    Parameters
    ----------
    facet_list (list of list) : Facet list that we want to convert to a bipartite graph
    Returns
    g (NetworkX graph object) : Bipartite graph with two node sets well identified
    facet_node_list (list of labeled nodes) : Set of nodes that represent facets in the bipartite graph
    -------
    """
    g = nx.Graph()
    facet_node_list = []
    for f, facet in enumerate(facet_list):
        g.add_node('f'+str(f), bipartite = 1)
        facet_node_list.append('f'+str(f))
        for v in facet:
            g.add_node('v'+ str(v), bipartite = 0)
            g.add_edge('v' + str(v), 'f' + str(f))  # differentiate node types
    return g, facet_node_list


def compute_betti(facetlist, highest_dim):
    """
    This function computes the betti numbers of the j-skeleton of the simplicial complex given in the shape of a facet
    list.
    Parameters
    ----------
    facetlist (list of list) : Sublists are facets with the index of the nodes in the facet.
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim.
    Returns
    -------
    The (highest_dim - 1) Betti numbers of the simplicial complexe.
    """

    st = gudhi.SimplexTree()
    for facet in facetlist:

        # This condition and the loop it contains break a facet in all its n choose k faces. This is more
        # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
        # dimension of the faces has to be at least one unit bigger than the skeleton we use.
        if len(facet) > highest_dim+1:
            for face in itertools.combinations(facet, highest_dim+1):
                st.insert(face)
        else:
            st.insert(facet)

    # We need to add a facet that has a size equivalent to the Betti we want to compute + 2 because GUDHI cannot compute
    # Betti N if there are no facets of size N + 2. As an example, if we take the 1-skeleton, there are, of course, no
    # facet of size higher than 2 and does not compute Betti 1. As a result, highest_dim needs to be 1 unit higher than
    # the maximum Betti number we want to compute. Moreover, we need to manually add a facet of size N + 2 (highest_dim + 1)
    # for the case where there are no such facets in the original complex. As an example, if we have two empty triangles,
    # GUDHI kinda assumes it's a 1-skeleton, and just compute Betti 0, although we would like to know that Betti 1 = 2.
    # The only way, it seems, to do so it to add a facet of size N + 2 (highest_dim + 1)
    # It also need to be disconnected from our simplicial complex because we don't want it to create unintentional holes.
    # This has the effect to add a componant in the network, hence why we substract 1 from Betti 0 (see last line of the function)

    # We add + 1 + 2 for the following reasons. First, we need a new facet of size highest_dim + 1 because of the reason
    # above. The last number in arange is not considered (ex : np.arange(0, 3) = [0, 1, 2], so we need to add 1 again.
    # Moreover, we cannot start at index 0 (because 0 is in the complex) nor -1 because GUDHI returns an error code 139.
    # If we could start at -1, it would be ok only with highest_dim + 3, but since we start at -2, we need to go one
    # index further, hence + 1.
    disconnected_facet = [label for label in np.arange(-2, -(highest_dim + 1 + 2), -1)]
    st.insert(disconnected_facet)


    # This function has to be launched in order to compute the Betti numbers
    st.persistence()
    bettis = st.betti_numbers()
    bettis[0] = bettis[0] - 1
    return bettis

def compute_and_store_bettis(path, highest_dim, save_path):
    """
    Computes and saves the bettis numbers of the j-skeleton (j = highest_dim) of a facetlist stored in a .txt file.
    Parameters
    ----------
    path (str) : Path to the facetlist stored in a txt file. The shape of the data correspond to the shape of the outputs
                 of the null_model.py module.
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim. See The Simplex Tree: An Efficient Data Structure
                        for General Simplicial Complexes by Boissonat, Maria for information about j-skeletons.
    save_path (str) : Path where to save the bettinumbers (npy file)
    Returns
    -------
    """
    bettilist = []
    facetlist = []
    print('Working on ' + path)
    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])

        if highest_dim == 'max':
            # If we want to compute Betti N, we need the (N+1)-skeleton
            # The highest dimensional facet that we want corresponds to the highest betti number we want to compute + 1
            # As an example, if we can only compute Betti 1, we need to use the 2-skeleton (because if we use the
            # 1-skeleton we transform full triangles in hollow triangles, which affects the 'True' number of holes.)
            highest_dim = highest_possible_betti(facetlist) + 1

        bettis = compute_betti(facetlist, highest_dim)

        # Some filtered facet list might only contain facets smaller than highest_dim. In this case, GUDHI does not compute
        # the highest Bettis we were expecting by setting highest_dim. In this case, however, the Bettis that were not
        # computed are necessarily zero, since the dimension of the facets does not allow the formation of higher dimensional
        # voids. For example, if there cannot be tetrahedrons, Betti 3 is necesserily zero, because we cannot glue tetrahedrons
        # together and create a 4 dimensional void. The following loop adds zeros to the Bettis that were not computed if
        # such a situation arises and ensures that there a no issues in the construction/dimensions of the returned numpy array.
        if len(bettis) < highest_dim:
            bettis.extend(0 for i in range(highest_dim - len(bettis)))
        bettilist.append(bettis)
        print(bettis)
        np.save(save_path + '_bettilist', np.array(bettilist))


def compute_and_store_bettis_from_instances(instance_path, idx_range, highest_dim, save_path):
    """
    Computes and saves the bettis numbers of the j-skeletons (j = highest_dim) of many facetlist generated with null_model.py
    and stored in .json files.
    Parameters
    ----------
    instance_path (str) : path to the instance, must not include the index of the instance nor its extension (added automatically)
    idx_range (range) : range of the indices of the instances
    highest_dim (int) : Highest dimension allowed for the facets. If a facet is of higher dimension in the facet list,
                        the algorithm breaks it down into it's N choose k faces. The resulting simplicial complexe is
                        called the j-skeleton, where j = highest_dim. See The Simplex Tree: An Efficient Data Structure
                        for General Simplicial Complexes by Boissonat, Maria for information about j-skeletons.
    save_path (str) : path where to save the array of betti numbers.
    Returns
    -------
    """

    bettilist = []
    highest_dim_param = highest_dim
    highest_dim_list = []
    for idx in idx_range:
        with open(instance_path + str(idx) + '.json', 'r') as file :
            print('Working on ' + instance_path + str(idx) + '.json')
            facetlist = json.load(file)
            if highest_dim_param == 'max':
                # If we want to compute Betti N, we need the (N+1)-skeleton
                # The highest dimensional facet that we want corresponds to the highest betti number we want to compute + 1
                # As an example, if we can only compute Betti 1, we need to use the 2-skeleton (because if we use the
                # 1-skeleton we transform full triangles in hollow triangles, which affects the 'True' number of holes.)
                highest_dim = highest_possible_betti(facetlist) + 1
                highest_dim_list.append(highest_dim)
                print(highest_dim)

            bettis = compute_betti(facetlist, highest_dim)
            # Some filtered facet list might only contain facets smaller than highest_dim. In this case, GUDHI does not compute
            # the highest Bettis we were expecting by setting highest_dim. In this case, however, the Bettis that were not
            # computed are necessarily zero, since the dimension of the facets does not allow the formation of higher dimensional
            # voids. For example, if there cannot be tetrahedrons, Betti 3 is necesserily zero, because we cannot glue tetrahedrons
            # together and create a 4 dimensional void. The following loop adds zeros to the Bettis that were not computed if
            # such a situation arises and ensures that there a no issues in the construction/dimensions of the returned numpy array.
            if highest_dim_param != 'max':
                if len(bettis) < highest_dim:
                    bettis.extend(0 for i in range(highest_dim - len(bettis)))
                bettilist.append(bettis)
                np.save(save_path + '_bettilist', np.array(bettilist))
            else:
                bettilist.append(bettis)
                for sublist in bettilist:
                    if len(sublist) < max(highest_dim_list):
                        sublist.extend(0 for i in range(max(highest_dim_list) - len(sublist)))
                print('Bettis : ', bettis)
                np.save(save_path + '_bettilist', np.array(bettilist))

def plot_betti_dist(bettiarray_instance, bettiarray_data):
    """
    This function plots the distributions of the betti numbers of the instances.
    Parameters
    ----------
    bettiarray_instance (np.array) : saved array using compute_and_store_bettis_from_instances
    bettiarray_data (np.array) : saved array using compute_and_store_bettis
    Returns
    -------
    """
    for column_index in range(0, bettiarray_instance.shape[1]):
        plt.figure(column_index)
        n, b, p = plt.hist(bettiarray_instance[:, column_index], bins=np.arange(0, max(bettiarray_instance[:, column_index]) + 0.5), density=True)
        plt.plot([bettiarray_data[0, column_index], bettiarray_data[0, column_index]], [0, max(n)], color="#ff5b1e")
        plt.text(-0.29, 25, 'Real system', color="#ff5b1e")
        # plt.ylim(0, 30)
        plt.xlabel('Number of Betti ' + str(column_index))
        plt.ylabel('Normalized count')

    plt.show()

def highest_possible_betti(facetlist):
    """
    This function computes the dimension of the highest non trivial Betti number in a facet list (by dimension we mean
    the index beside a Betti number, for example Betti 2 has a ' dimension ' 2, although it corresponds to a 3 dimensional
    void). It is usefull since, by default, GUDHI's simplex tree tend to compute the Betti numbers up to the
    dimension - 1 of the highest dimensional facet in the list. As an example, if there is only one facet of size 38
    (dimension 37), GUDHI tries to compute the Betti numbers from Betti 0 to Betti 36, which is unnecessary since there
    is just ONE facet in our simplicial complex meaning that every Betti number higher than Betti 0 is trivial 0
    (there cannot be holes!).
    Parameters
    ----------
    facetlist (List of lists) : List that contains sublist. These sublists contain the indices of the nodes that are part
                                of the facet (each sublist is a maximal facet).
    Returns
    -------
    The dimension of the highest non trivial Betti number.
    """

    # Sort the facet list by ascending order of facet size
    facetlist.sort(key=len)

    # This loop counts the number of facets of size 1 (meaning 0-simplices)
    i = 0
    while i < len(facetlist):
        if len(facetlist[i]) != 1:
            break
        i += 1
    # If we only have facets of size 1, there are no facets bigger than 1
    if i == len(facetlist):
        facets_bigger_than_one = []
    # If the previous loop broke at i, it means that all the other facets have size greater than 1.
    else:
        facets_bigger_than_one = facetlist[i:]

    # Number of facets that have size bigger than 1
    length = len(facets_bigger_than_one)

    # If we have a minimum of 3 facets bigger than one, we can, in theory, construct holes. As an example, as soon as we
    # have 3 1-simplices, we can build an empty triangle.
    if length >= 3:
        min_size = len(facets_bigger_than_one[0])
        max_size = len(facets_bigger_than_one[-1])

        nb_facets_histogram_by_size = []

        # Initialize a list that will contain each unique sizes
        size_list = []
        # Initialize a list that will contain the number of facets of each size
        count_list = []
        previous_facet = facets_bigger_than_one[0]
        i = 1
        count = 0
        # This loop counts the number of facets of the same size and stores them in a list [size, count] that is
        # also stored in a list (nb_facets_histogram_by_size)
        while i < length:
            present_facet = facets_bigger_than_one[i]
            if len(previous_facet) == len(present_facet):
                count += 1
            else:
                count += 1
                nb_facets_histogram_by_size.append([len(previous_facet), count])
                size_list.append(len(previous_facet))
                count_list.append(count)
                count = 0
            previous_facet = present_facet
            i += 1
        count += 1
        nb_facets_histogram_by_size.append([len(previous_facet), count])
        size_list.append(len(previous_facet))
        count_list.append(count)
        size_list = np.array(size_list)
        count_list = np.array(count_list)


        # If there are N facets, we can, in theory, have a non zero Betti N-2, provided that these N facets have
        # a minimum size of N - 1. Ex : 4 facets, could build an empty tetrahedron IF their size is at least 3.
        # Therefore : If the maximal size that we can find in the list is lower than the minimum size (i.e. lower than
        # N-1) required to have a non zero Betti N - 2, we know that Betti >= N - 2 are 0. This also means that
        # all the Betti numbers > max_size - 1 are zero. For instance, if we have 10 facets, we can, in theory build
        # a hole that would contribute to Betti N - 2 = 8. But if the largest facet in the list has size 4, we know that
        # we cannot build holes that would contribute to Betti > 3. Hence Betti > 3 are all zero.
        # Indeed, to build Betti numbers of dimension 'd', we need facets of size d + 1. So if our max size is 'k' the
        # first Betti number we need to look at is k - 1 (i.e. max_size - 1)

        # if max_size < N - 1 would be another way to write this.
        if max_size < length - 1:
            considered_size = max_size

        # If there are facets of size higher than or equal to the minimum size required to have a non zero Betti N - 2,
        # the first Betti number we need to look at is built with facets of size N - 1 and bigger. This means that
        # we don't have to look at the Betti numbers > N - 2, that would in principle be non trivial if we didn't know
        # the number of facets
        else:
            # considered_size = N - 1 would be another way to write this.
            considered_size = length - 1

        # This loop checks if there is a sufficient number of facets of a specific size (and higher than this specific size)
        # to create the associated Betti number. The first size that respects this condition corresponds to our highest non
        # trivial Betti number.
        # We need to iterate over every size below the first ' considered size ' because we can encounter a situation
        # where there are no facets of a certain size, but enough facets of higher size to create a betti number associated
        # with this size. For example, in a facet list in which there is 1 facet of size 2, 3 facets if size 4 and 1 facet
        # of size 5, we know that Betti 3 and Betti 2 are zero, but Betti 1 could be non zero since there are 5 facets
        # of size >= 2.
        for size in np.arange(considered_size, 1, - 1):
            nb_of_respecting_facets = np.sum(count_list[np.where(size_list >= size)])
            if nb_of_respecting_facets >= size + 1:
                break
        betti = size - 1

    # TODO : This 2 seems out of place, it should be Betti = 0. Test
    else:
        betti = 2

    return betti


def build_simplex_tree(facetlist, highest_dim):
    st = gudhi.SimplexTree()
    for facet in facetlist:

        # This condition and the loop it contains break a facet in all its n choose k faces. This is more
        # memory efficient than inserting a big facet because of the way GUDHI's SimplexTree works. The
        # dimension of the faces has to be at least one unit bigger than the skeleton we use.
        if len(facet) > highest_dim + 1:
            for face in itertools.combinations(facet, highest_dim + 1):
                st.insert(face)
        else:
            st.insert(facet)

    return st



def skeleton_to_graph(simplex_tree):

    g = nx.Graph()

    for simplex in simplex_tree.get_skeleton(1):
        if len(simplex[0]) == 2:
            g.add_edge(*simplex[0])

    return g



def compute_nb_simplices(simplex_tree, dim, andlower=True):

    i = 0
    if andlower:
        while dim - i >= 0:
            count = 0

            for simplex in simplex_tree.get_skeleton(dim - i):
                if len(simplex[0]) == (dim - i) +1:
                    count += 1
            print('Number of ' + str(dim-i) + '-simplices : ', count)

            i += 1
    else:
        count = 0

        for simplex in simplex_tree.get_skeleton(dim):
            if len(simplex[0]) == dim + 1:
                count += 1
        print('Number of ' + str(dim - i) + '-simplices : ', count)
        return count
    return

    # print(len(st.get_skeleton(1)))

def build_facet_list(file_name, matrix, one_simplices_file, two_simplices_file, three_simplices_file, alpha):
    #open('facet_list.txt', 'a').close()

    with open(file_name, 'w') as facetlist:

        nodelist = np.arange(0, matrix.shape[0])

        for node in nodelist:
            facetlist.write(str(node) +'\n')

        with open(one_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + '\n')
                except:
                    pass

        with open(two_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + '\n')
                except:
                    pass
        with open(three_simplices_file, 'r') as csvfile:

            reader = csv.reader(csvfile)
            next(reader)

            for row in tqdm(reader):

                try:
                    p = float(row[-1])
                    if p < alpha:
                        # Reject H_0 in which we suppose that u and v are independent
                        # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                        facetlist.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3]) + '\n')
                except:
                    pass
    return


if __name__ == '__main__':

    data_name = 'opsahl-ucforum'
    dir_name = 'results'
    df = pd.read_csv('KONECT_data\out.' + data_name, sep=' ', skiprows=2, header=None)

    k = pd.crosstab(df[0], df[1])

    data_matrix = k.to_numpy()

    pd_matrix = pd.DataFrame(data_matrix)
    p, n = data_matrix.shape
    print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

    onesimp_file = os.path.join(dir_name, data_name + '_asymptotic_pvalues.csv')
    twosimp_file = os.path.join(dir_name, data_name + '_asymptotic_2-simplices_01.csv')
    threesimp_file = os.path.join(dir_name, data_name + '_3-simplices.csv')
    build_facet_list(data_name + '_facetlist.txt', data_matrix, onesimp_file, twosimp_file, threesimp_file, 0.01)



    path = data_name + '_facetlist.txt'
    facetlist = []
    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])
    somme = 0
    maxlist =[]
    for elem in facetlist:
        maxlist.append(len(elem))
        somme += len(elem)
    print(somme/len(facetlist))
    print(min(maxlist), max(maxlist))

    st = build_simplex_tree(facetlist, 3)
    compute_nb_simplices(st, 3)

    print(compute_betti(facetlist, 4))

    exit()

    data_name = 'web_of_life'
    dir_name = 'results'

    data_matrix = np.loadtxt(open(r'M_PL_062.csv', 'r'), delimiter=',')

    data_matrix = (data_matrix > 0) * 1

    data_matrix = data_matrix.T

    p, n = data_matrix.shape
    print("Preparing to analyze a data set with " + str(p) + " variables and " + str(n) + " samples")

    onesimp_file = os.path.join(dir_name, data_name + '_asymptotic_pvalues.csv')
    twosimp_file = os.path.join(dir_name, data_name + '_asymptotic_2-simplices_01.csv')
    threesimp_file = os.path.join(dir_name, data_name + '_3-simplices.csv')
    build_facet_list(data_name + '_facetlist.txt', data_matrix, onesimp_file, twosimp_file, threesimp_file, 0.01)



    path = data_name + '_facetlist.txt'
    facetlist = []
    with open(path, 'r') as file:
        for l in file:
            facetlist.append([int(x) for x in l.strip().split()])
    somme = 0
    maxlist =[]
    for elem in facetlist:
        maxlist.append(len(elem))
        somme += len(elem)
    print(somme/len(facetlist))
    print(min(maxlist), max(maxlist))

    st = build_simplex_tree(facetlist, 3)
    compute_nb_simplices(st, 3)

    print(compute_betti(facetlist, 4))

    exit()