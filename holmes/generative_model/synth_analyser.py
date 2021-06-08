import copy
import networkx as nx
from tqdm import tqdm
import numpy as np
import itertools
from scipy.stats import chi2
from .loglin_model import *
import pickle
#from .Exact_chi_square_1_deg import *
from numba import jit


def pvalue_AB_AC_BC(cont_cube):
    """
    Find the p-value for a 2X2X2 contingency cube and the model of no second order interaction
    :param cont_cube: (np.array of ints) 2X2X2 contingency cube
    :return: If the expected table under the model can be computed, the p-value is returned. Otherwise 'None' is returned
    """
    expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
    if expected is not None:
        return chisq_test(cont_cube, expected)[1]
    else:
        return expected


def to_occurrence_matrix(matrix, savepath=None):
    """
    Transform a matrix into a binary matrix where entries are 1 if the original entry was different from 0.
    Parameters
    ----------
    matrix (np.array)
    savepath (string) : path and filename under which to save the file
    Returns
    -------
        The binary matrix or None if a savepath is specified.
    """
    if savepath is None:
        return (matrix > 0) * 1
    else:
        np.save(savepath, (matrix > 0) * 1)

@jit(nopython=True)
def get_cont_table(u_idx, v_idx, matrix):
    """
    Computes the 2X2 contingency table for two elements in the presence/absence matrix

    :param u_idx: (int) index of the row associated to the element u
    :param v_idx: (int) index of the row associated to the element v
    :param matrix: (np.array of ints) Presence/absence matrix
    :return: (np.array of ints) 2X2 contingency table of the elements u and v
    """

    # u present, v present
    table11 = 0

    # u present, v NOT present
    table10 = 0

    # u NOT present, v present
    table01 = 0

    # u NOT present, v NOT present
    table00 = 0

    for i in range(0, matrix.shape[1]):
        u_state = matrix[u_idx, i]
        v_state = matrix[v_idx, i]

        if u_state == 0:
            if v_state == 0:
                table00 += 1
            else:
                table01 += 1
        else:
            if v_state == 0:

                table10 += 1
            else:
                table11 += 1

    return np.array([[table00, table01], [table10, table11]])


@jit(nopython=True)
def get_cont_cube(u_idx, v_idx, w_idx, matrix):
    """
    Computes the 2X2X2 contingency cube for three elements in the presence/absence matrix

    :param u_idx: (int) index of the row associated to the element u
    :param v_idx: (int) index of the row associated to the element v
    :param w_idx: (int) index of the row associated to the element w
    :param matrix: (np.array of ints) Presence/absence matrix
    :return: (np.array of ints) 2X2X2 contingency cube of the elements u, v and w
    """
    # Computes the 2X2X2 contingency table for the occurrence matrix

    # All present :
    table000 = 0

    # v absent
    table010 = 0

    # u absent
    table100 = 0

    # u absent, v absent
    table110 = 0

    # w absent
    table001 = 0

    # v absent, w absent
    table011 = 0

    # u absent, w absent
    table101 = 0

    # all absent
    table111 = 0
    for i in range(0, matrix.shape[1]):
        u_state = matrix[u_idx, i]
        v_state = matrix[v_idx, i]
        w_state = matrix[w_idx, i]
        if u_state == 0:
            if v_state == 0:
                if w_state == 0:
                    table000 += 1
                else:
                    table100 += 1
            else:
                if w_state == 0:
                    table001 += 1
                else:
                    table101 += 1
        else:
            if v_state == 0:
                if w_state == 0:
                    table010 += 1
                else:
                    table110 += 1
            else:
                if w_state == 0:
                    table011 += 1
                else:
                    table111 += 1

    return np.array([[[table000, table010], [table100, table110]], [[table001, table011], [table101, table111]]],
                    dtype=np.float64)


def chisq_test(cont_tab, expected, df=1):
    """
    Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    via MLE or iterative proportional fitting.
    :param cont_tab: (np.array of ints) 2X2 contingency table
    :param expected: (np.array of ints) 2X2 contingency table of expected values under the independance model
    :param df: (int) Degrees of freedom of the chi^2 distribution (for asymptotic tests, it should be 1 regardless of
                     the number of variables. Note that this is only true if the table only contains binary variables.)
    :return: The chi^2 test statistics (float between 0 and inf) and the p-value (float between 1 and 0).
    """

    if float(0) in expected:
        test_stat = 0
        p_val = 1
    else:
        test_stat = test_statistics(cont_tab, expected)
        p_val = chi2.sf(test_stat, df)

    return test_stat, p_val


@jit(nopython=True)
def test_statistics(cont_tab, expected):
    """
    Computes (using numba) the chi^2 statistics between a contingency table and the expected table under a given model.
    :param cont_tab: (np.array of ints) 2X2 contingency table
    :param expected: (np.array of ints) 2X2 contingency table of expected values under the independance model
    :return: The chi^2 test statistics (float between 0 and inf)
    """
    teststat = 0
    cont_tab = cont_tab.flatten()
    expected = expected.flatten()

    for i in range(len(cont_tab)):
        teststat += (cont_tab[i] - expected[i]) ** 2 / expected[i]

    return teststat


def save_all_triangles(G):
    """
    Find all triangles (cliques of size 3) in a (NetworkX) graph. This is a necessary function for the 'step method'
    :param G: NetworkX graph
    :param savename: (str) Name of the csv file where we save information
    :param bufferlimit: (int) Save every 'bufferlimit' triangles found. (With the parameter, we avoid saving each and
                              every time a triangle is found).
    :return: None
    """
    G = copy.deepcopy(G)

    # Iterate over all possible triangle relationship combinations
    triangle_set = set()
    for node in list(G.nodes):
        if G.degree[node] < 2:
            G.remove_node(node)
        else:
            for n1, n2 in itertools.combinations(G.neighbors(node), 2):

                # Check if n1 and n2 have an edge between them
                if G.has_edge(n1, n2):
                    triangle_set.add((node, n1, n2))

            G.remove_node(node)

    return triangle_set

def find_unique_tables(matrix):
    """
    Find the unique contingency tables in the dataset. Sometimes, the number of of unique contingency tables is lower
    than the number of pairs we can create. Thus, we can save time be first running this function (especially if there
    are many pairs we can create).
    :param matrix: (np.array of ints) Presence/absence matrix
    :param save_name: (str) Name of the csv file where we save information
    :return:
    """

    table_set = set()

    # Finds all unique tables

    for one_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 2)):

        computed_cont_table = get_cont_table(one_simplex[0], one_simplex[1], matrix)
        # computed_cont_table = computed_cont_table.astype(int)

        table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
            computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])

        if table_str not in table_set:
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different tables : ', len(table_set))
    #json.dump(table_set, open(save_name + "_table_list.json", 'w'))
    return table_set

def pvalues_for_tables(table_set):
    """
    Find the p-values for the unique contingency tables found with the function find_unique_tables
    :param file_name: (str) Path to the file obtained with the function find_unique_tables. This also acts as a
                            savename for the dictionary table : (chi^2 statistics, p-value). The keys of the dictionary
                            are actually strings where we flattened the 2X2 contingency table and separate each entry
                            by underscores.
    :return: None
    """

    #with open(file_name + "_table_list.json") as json_file:
    #    table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

    pvaldictio = {}

    # Max index used in range() :
    for it in tqdm(range(len(table_set))):
        table_id = table_set[it]
        table = np.random.rand(2, 2)
        table_id_list = str.split(table_id, '_')
        table[0, 0] = int(table_id_list[0])
        table[0, 1] = int(table_id_list[1])
        table[1, 0] = int(table_id_list[2])
        table[1, 1] = int(table_id_list[3])

        expected1 = mle_2x2_ind(table)
        pvaldictio[table_id] = chisq_test(table, expected1, df=1)

    #json.dump(pvaldictio, open(file_name + "_asymptotic_pval_dictio.json", 'w'))

    return pvaldictio


def pairwise_p_values(bipartite_matrix, dictionary, alpha):
    """
    Find the p-values of pairs of elements in the presence/absence matrix. To run this function, we need to find the
    unique tables with pvalues_for_tables.
    :param bipartite_matrix: (np.array of ints) Presence/absence matrix
    :param dictionary: (dictionary) Dictionary obtained with the function pvalues_for_tables. This dictionary has the
                                    format table : (chi^2 statistics, p-value). The keys of the dictionary are actually
                                    strings representing the entry of a flattened contingency table separated by
                                    underscores
    :return: None
    """

    graph = nx.Graph()

    link_set = set()
    for one_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 2)):
        contingency_table = get_cont_table(one_simplex[0], one_simplex[1], bipartite_matrix)
        table_str = str(contingency_table[0, 0]) + '_' + str(contingency_table[0, 1]) + '_' + \
                    str(contingency_table[1, 0]) + '_' + str(contingency_table[1, 1])

        chi2, p = dictionary[table_str]

        try:
            if p < alpha:
                link_set.add(one_simplex)
                graph.add_edge(int(one_simplex[0]), int(one_simplex[1]), p_value=p)
        except:
            pass

    return link_set, graph

def find_unique_cubes(matrix):
    """
    Find the unique contingency cubes in the dataset. Sometimes, the number of of unique cubes is lower
    than the number of triplets we can create. Thus, we can save time be first running this function (especially if there
    are many triplets we can create).
    :param matrix: (np.array of ints) Presence/absence matrix
    :param save_name: (str) Name of the csv file where we save information
    :return: None
    """

    table_set = set()

    for two_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 3)):
        cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix)

        if not find_if_invalid_cube(cont_cube):
            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different valid cubes : ', len(table_set))

    return table_set

# @jit(nopython=True)
def find_if_invalid_cube(cont_cube):
    """
    Function used to know whether a sufficient configuration contains a zero (which indicates an invalid table).
    :param cont_cube: (np.array of ints) 2X2X2 contingency cube.
    :return: 1 if the table is invalid, 0 otherwise.
    """
    xij_ = np.sum(cont_cube, axis=0)
    nonzeroij = np.count_nonzero(xij_)
    if nonzeroij != 4:
        return 1
    xi_k = np.sum(cont_cube, axis=2)
    nonzeroik = np.count_nonzero(xi_k)
    if nonzeroik != 4:
        return 1
    x_jk = np.sum(cont_cube, axis=1).T
    nonzerojk = np.count_nonzero(x_jk)
    if nonzerojk != 4:
        return 1

    return 0


def pvalues_for_cubes(table_set):
    """
    Find the p-values for the unique contingency cubes found with the function find_unique_cubes
    :param
    :return: None
    """


    #### From the different tables : generate the chisqdist :

    pvaldictio = {}

    for it in tqdm(range(len(table_set))):

        table_id = table_set[it]
        table = np.random.rand(2, 2, 2)
        table_id_list = str.split(table_id, '_')
        table[0, 0, 0] = int(table_id_list[0])
        table[0, 0, 1] = int(table_id_list[1])
        table[0, 1, 0] = int(table_id_list[2])
        table[0, 1, 1] = int(table_id_list[3])
        table[1, 0, 0] = int(table_id_list[4])
        table[1, 0, 1] = int(table_id_list[5])
        table[1, 1, 0] = int(table_id_list[6])
        table[1, 1, 1] = int(table_id_list[7])

        expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)

        if expected is not None:

            pvaldictio[table_id] = chisq_test(table, expected, df=1)

        else:
            pvaldictio[table_id] = str(expected)

    return pvaldictio


def save_triplets_p_values_dictionary(bipartite_matrix, dictionary, savename):
    """
    Find the p-values of triplets of elements in the presence/absence matrix. To run this function, we need to find the
    unique cubes with pvalues_for_cubes.
    :param bipartite_matrix: (np.array of ints) Presence/absence matrix
    :param dictionary: (dictionary) Dictionary obtained with the function pvalues_for_cubes. This dictionary has the
                                    format table : (chi^2 statistics, p-value). The keys of the dictionary are actually
                                    strings representing the entry of a flattened contingency table separated by
                                    underscores
    :param savename: (str) Name of the csv file where we save information
    :return: None
    """

    # create a CSV file
    with open(savename + '.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])

        for two_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 3)):
            cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], bipartite_matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))

            try:
                chi2, p = dictionary[table_str]
            except:
                # TODO Change for None?
                chi2, p = 0.0, 1.0

            writer.writerow([two_simplex[0], two_simplex[1], two_simplex[2], p])

def triangles_p_values_tuple_dictionary(triangle_set, dictionary, matrix, alpha):
    """
    Fetch the p-values of triplets that form a triangle after the first step of the method.
    :param csvfile: (str) Path to the file obtained with the function save_triplets_p_values_dictionary.
    :param savename: Name of the csv file where we save information
    :param dictionary: (dictionary) Dictionary obtained with the function pvalues_for_cubes. This dictionary has the
                                    format table : (chi^2 statistics, p-value). The keys of the dictionary are actually
                                    strings representing the entry of a flattened contingency table separated by
                                    underscores
    :param matrix: (np.array of ints) Presence/absence matrix
    :return: None
    """

    two_simplices = set()
    for triangle in triangle_set:

        cont_cube = get_cont_cube(triangle[0], triangle[1], triangle[2], matrix)

        table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
            int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
            int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
            int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))
        try:
            chi2, p = dictionary[table_str]
            if p < alpha :
                two_simplices.add(triangle)

        except:
            pass

    return two_simplices

if __name__ == '__main__':


    #with open(r'dictionary_of_truth.pkl', 'rb') as dot_file:
    #    dot = pickle.load(dot_file)

    #exit()


    #Load the reference factorgraph and the reference for simplices:

    with open(r'D:\Users\Xavier\Documents\HOLMES\HOLMES\doc\sc_20_nodes\data_none_fg.pkl', 'rb') as fg_file:
        factorgraph = pickle.load(fg_file)

    effective_facet_list_dictio = factorgraph.effective_facet_list

    effective_one_simps = effective_facet_list_dictio[2]
    effective_two_simps = effective_facet_list_dictio[3]

    # Choose the alpha parameter to use throughout the analysis.
    alpha = 0.01

    dictionary_of_truth = {'vp_one' : [], 'fn_one' : [], 'fp_one' : [], 'vp_two' : [], 'fn_two' : [], 'fp_two' : []}

    for i in range(2):


        # Choose the name of the directory (dirName) where to save the files and the 'prefix' name of each created files
        # (data_name)
        #dirName = 'Directory'
        #data_name = 'Data'



        # Enter the path to the presence/absence matrix :
        matrix1 = np.load(r'D:\Users\Xavier\Documents\HOLMES\HOLMES\doc\sc_20_nodes\10000\data_none_bipartite_' + str(i) + '.npy')
        matrix1 = matrix1.astype(np.int64)

        ## Create target Directory if don't exist
        #if not os.path.exists(dirName):
        #    os.mkdir(dirName)
        #    print("Directory ", dirName, " Created ")
        #else:
        #    print("Directory ", dirName, " already exists")

        #data_name = os.path.join(dirName, data_name)

        ########## First step : Extract all the unique tables

        print('Step 1 : Extract all the unique tables')

        # Finds all unique tables
        all_unique_tables = find_unique_tables(matrix1)

        ######### Second step : Extract all the pvalues with an asymptotic distribution

        print('Step 2: Extract pvalues for all tables with an asymptotic distribution')

        pvaldictio_for_tables = pvalues_for_tables(all_unique_tables)

        ######### Third step : Find table for all links and their associated pvalue

        print('Step 3 : Find table for all links and their associated pvalue')

        #with open(data_name + '_asymptotic_pval_dictio.json') as jsonfile:
        #    dictio = json.load(jsonfile)

        ######### Fourth step : Choose alpha and extract the network
        print('Step 4 : Generate network and extract edge_list for a given alpha')
        link_set, g = pairwise_p_values(matrix1, pvaldictio_for_tables, alpha)

        print('Number of nodes : ', g.number_of_nodes())
        print('Number of links : ', g.number_of_edges())

        ######### Fifth step : Extract all the unique cubes

        print('Step 5 : Extract all the unique valid cubes')

        unique_cube_set = find_unique_cubes(matrix1)

        ######### Sixth step : Extract pvalues for all cubes with an asymptotic distribution

        print('Step 6: Extract pvalues for all cubes with an asymptotic distribution')

        pvaldictio_for_cubes = pvalues_for_cubes(unique_cube_set)

        ######## Seventh step : Find cube for all triplets and their associated pvalue

        print('Step method : ')

        ######## Fifth step : Find all triangles in the previous network

        print('Step 7 : Finding all empty triangles in the network')

        triangle_set = save_all_triangles(g)

        print('Number of triangles : ', len(triangle_set))

        ######## Sixth step : Find all the p-values for the triangles under the hypothesis of homogeneity

        print('Step 8 : Find all the p-values for the triangles under the hypothesis of homogeneity')


        two_simplices = triangles_p_values_tuple_dictionary(triangle_set, pvaldictio_for_cubes, matrix1, alpha)

        ######## Fifth step : Exctract all 2-simplices

        print('Extracted 2-simplices : \n', two_simplices)

        vp_one = effective_one_simps.intersection(link_set)
        fn_one = effective_one_simps - link_set
        fp_one = link_set - effective_one_simps

        vp_two = effective_two_simps.intersection(two_simplices)
        fn_two = effective_two_simps - two_simplices
        fp_two = two_simplices - effective_two_simps

        dictionary_of_truth['vp_one'].append(vp_one)
        dictionary_of_truth['fn_one'].append(fn_one)
        dictionary_of_truth['fp_one'].append(fp_one)

        dictionary_of_truth['vp_two'].append(vp_two)
        dictionary_of_truth['fn_two'].append(fn_two)
        dictionary_of_truth['fp_two'].append(fp_two)

    with open('dictionary_of_truth.pkl', 'wb') as f:
        pickle.dump(dictionary_of_truth, f)

        # THIS ONE GIVES ALL TRIANGLES THAT CONVERGED REGARDLESS OF THEIR P-VALUE (NO ALPHA NEEDED)
        # extract_converged_triangles(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', data_name + '_converged_triangles')

        ################# Comparison ###################


