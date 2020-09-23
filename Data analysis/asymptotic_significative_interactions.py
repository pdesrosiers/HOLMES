import copy
import networkx as nx
from tqdm import tqdm
from Exact_chi_square_1_deg import *
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

def pvalue_ABC_ABD_ACD_BCD(hyper_cont_cube):
    """
    Find the p-value for a 2X2X2X2 hyper contingency cube and the model of no third order interaction
    :param hyper_cont_cube: (np.array of ints ) 2X2X2X2 hyper contingency cube
    :return: If the expected table under the model can be computed, the p-value is returned. Otherwise 'None' is returned
    """
    expected = ipf_ABC_ABD_ACD_BCD_no_zeros(hyper_cont_cube)
    if expected is not None:
        return chisq_test(hyper_cont_cube, expected)[1]
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
        return (matrix > 0)*1
    else:
        np.save(savepath, (matrix>0)*1)

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

    #All present :
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

    return np.array([[[table000, table010], [table100, table110]], [[table001, table011], [table101, table111]]], dtype=np.float64)


def phi_coefficient_table(cont_tab):
    """
    Computes the phi coefficient for a given contingency table
    :param cont_tab: (np.array of ints) 2X2 contingency table
    :return: (float) phi coefficient between -1 and 1
    TODO We could also use another formula using the chi^2 statistics. The following function might cause overflows
    (see function below)
    """
    row_sums = np.sum(cont_tab, axis=1)
    col_sums = np.sum(cont_tab, axis=0)
    return (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1])/np.sqrt(row_sums[0]*row_sums[1]*col_sums[0]*col_sums[1])

def phi_coefficient_chi(cont_tab, chi):
    #TODO we need to implement the +/- sign before the square root. We could use a reduced numerator like
    # (cont_tab[0,0]*cont_tab[1,1] - cont_tab[1,0]*cont_tab[0,1]) / FACTOR TO BE DETERMINED TO AVOID OVERFLOWS
    n = np.sum(cont_tab)

    return np.sqrt(chi/n)

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

def read_pairwise_p_values(filename, alpha=0.01):
    """
    Creates a NetworkX graph from a csv file created with save_pairwise_p_values_phi_dictionary.
    We only include nodes that have at least one edge (meaning that we rejected independence between this node and
    another one.)
    :param filename: (str) Path to the csv file.
    :param alpha:  (float) Threshold of significance
    :return: NetworkX graph
    """

    graph = nx.Graph()

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile)
        next(reader)

        for row in tqdm(reader):

            try:
                p = float(row[-1])
                if p < alpha:
                    # Reject H_0 in which we suppose that u and v are independent
                    # Thus, we accept H_1 and add a link between u and v in the graph to show their dependency
                    graph.add_edge(int(row[0]), int(row[1]), phi=float(row[-2]), p_value=p)
            except:
                pass

    return graph

def save_all_triangles(G, savename, bufferlimit=100000):
    """
    Find all triangles (cliques of size 3) in a (NetworkX) graph. This is a necessary function for the 'step method'
    :param G: NetworkX graph
    :param savename: (str) Name of the csv file where we save information
    :param bufferlimit: (int) Save every 'bufferlimit' triangles found. (With the parameter, we avoid saving each and
                              every time a triangle is found).
    :return: None
    """
    G = copy.deepcopy(G)
    with open(savename + '.csv', 'w',  newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3']])
    buffer = []
    # Iterate over all possible triangle relationship combinations
    count = 0
    for node in list(G.nodes):
        if G.degree[node] < 2:
            G.remove_node(node)
        else:
            for n1, n2 in itertools.combinations(G.neighbors(node), 2):

                # Check if n1 and n2 have an edge between them
                if G.has_edge(n1, n2):

                    buffer.append([node, n1, n2, G.get_edge_data(node, n1)['phi'],
                                   G.get_edge_data(node, n2)['phi'], G.get_edge_data(n1, n2)['phi']])
                    count += 1

            G.remove_node(node)

            if count == bufferlimit:
                with open(savename + '.csv', 'a',  newline='') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(buffer)
                    count = 0
                    # empty the buffer
                    buffer = []

    with open(savename + '.csv', 'a',  newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def triangles_p_values_AB_AC_BC(csvfile, savename, matrix, bufferlimit=100000):
    """
    Computes the p-value for the model of no second order interaction between triplets that form a triangle after applying
    the first step of the step method. (In order to use this function, we need to run save_all_triangles first).
    :param csvfile: (str) Path to the CSV files obtained with the function save_all_triangles
    :param savename: (str) Name of the csv file where we save information
    :param matrix: (np.array of ints) Presence/absence matrix
    :param bufferlimit: (int) Save every 'bufferlimit' triangles found. (With the parameter, we avoid saving each and
                              every time a triangle is found).
    :return: none_count (int) number triplets we could not evaluate because the contingency cube has inexistent MLE.
    """

    buffer = []

    with open(csvfile, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        count = 0
        none_count = 0
        next(reader)
        for row in tqdm(reader):

            cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

            p_value = pvalue_AB_AC_BC(cont_cube)

            if p_value is None:
                none_count += 1
            buffer.append([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p_value])
            count += 1

            if count == bufferlimit:
                writer.writerows(buffer)
                count = 0
                # empty the buffer
                buffer = []


        writer.writerows(buffer)
        return none_count

def count_triangles_csv(filename):
    """
    Count the number of triangles (cliques of size 3) in a csv file generated with save_all_triangles.
    :param filename: (str) Path to the CSV files obtained with the function save_all_triangles
    :return: (int) Number of triangles
    TODO Obsolete / could be moved somewhere else
    """
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row_count = -1
        for row in tqdm(reader):
            row_count +=1

    return row_count

def save_triplets_p_values(bipartite_matrix, savename):

    """
    Computes the p-value for the model of no second order interaction between triplets. This function is used in the
    systematic method. (In order to use this function, we need to run save_all_triangles first).
    :param bipartite_matrix:(np.array of ints) Presence/absence matrix
    :param savename:(str) Name of the csv file where we save information
    :return: None
    """

    # create a CSV file
    with open(savename+'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])

        for two_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 3)):
            cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], bipartite_matrix)


            p_value = pvalue_AB_AC_BC(cont_cube)

            writer = csv.writer(csvFile)
            writer.writerow([two_simplex[0], two_simplex[1], two_simplex[2], p_value])

def extract_converged_triangles(csvfilename, savename):
    """
    Find all triplets that we were able to test (meaning that we were able to compute a p-value for the model of no
    second order interaction). This function essentially filters out triplets that did not produce a p-value.
    :param csvfilename:(str) Path to the CSV files obtained with the function save_all_triangles
    :param savename: str) Name of the csv file where we save information
    :return: None
    """
    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'phi 1-2', 'phi 1-3', 'phi 2-3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[6])
                writer = csv.writer(fout)
                writer.writerow([int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), p])
            except:
                pass


def extract_phi_for_triangles(csvfilename):
    """
    Find how many triangles present only pairwise positive/negative interactions or a mix of positive and negative
    interactions
    :param csvfilename: (str) Path to the CSV files obtained with the function triangles_p_values_AB_AC_BC
    :return: (list of ints) number of triangles where each pairwise interaction is negative, where there is one positive
                            interaction and two negative interactions, where there are two positive and one negative and
                            where they are all positive.
    TODO Obsolete / could be moved somewhere else
    """
    with open(csvfilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        pure_negative_count = 0
        pure_positive_count = 0
        one_pos_two_neg = 0
        two_pos_one_neg = 0

        for row in reader:
            try:
                philist = [float(row[3]), float(row[4]), float(row[5])]
                philistmask = (np.array(philist) > 0) * 1
                sum = np.sum(philistmask)
                if sum == 3:
                    pure_positive_count += 1
                elif sum == 0:
                    pure_negative_count += 1
                elif sum == 1:
                    one_pos_two_neg += 1
                else:
                    two_pos_one_neg += 1
            except:
                pass

    return [pure_negative_count, one_pos_two_neg, two_pos_one_neg, pure_positive_count]

def find_unique_tables(matrix, save_name):
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
        #computed_cont_table = computed_cont_table.astype(int)

        table_str = str(computed_cont_table[0, 0]) + '_' + str(computed_cont_table[0, 1]) + '_' + str(
            computed_cont_table[1, 0]) + '_' + str(computed_cont_table[1, 1])

        if table_str not in table_set:
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different tables : ', len(table_set))
    json.dump(table_set, open(save_name + "_table_list.json", 'w'))

def pvalues_for_tables(file_name):
    """
    Find the p-values for the unique contingency tables found with the function find_unique_tables
    :param file_name: (str) Path to the file obtained with the function find_unique_tables. This also acts as a
                            savename for the dictionary table : (chi^2 statistics, p-value). The keys of the dictionary
                            are actually strings where we flattened the 2X2 contingency table and separate each entry
                            by underscores.
    :return: None
    """

    with open(file_name + "_table_list.json") as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        # Max index used in range() :
        for it in tqdm(range(len(table_set))):
            table_id = table_set[it]
            table = np.random.rand(2,2)
            table_id_list = str.split(table_id, '_')
            table[0, 0] = int(table_id_list[0])
            table[0, 1] = int(table_id_list[1])
            table[1, 0] = int(table_id_list[2])
            table[1, 1] = int(table_id_list[3])

            expected1 = mle_2x2_ind(table)
            pvaldictio[table_id] = chisq_test(table, expected1, df=1)

        json.dump(pvaldictio, open(file_name + "_asymptotic_pval_dictio.json", 'w'))

def save_pairwise_p_values_phi_dictionary(bipartite_matrix, dictionary, savename):
    """
    Find the p-values of pairs of elements in the presence/absence matrix. To run this function, we need to find the
    unique tables with pvalues_for_tables.
    :param bipartite_matrix: (np.array of ints) Presence/absence matrix
    :param dictionary: (dictionary) Dictionary obtained with the function pvalues_for_tables. This dictionary has the
                                    format table : (chi^2 statistics, p-value). The keys of the dictionary are actually
                                    strings representing the entry of a flattened contingency table separated by
                                    underscores
    :param savename: (str) Name of the csv file where we save information
    :return: None
    """

    # create a CSV file
    with open(savename+'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'phi-coefficient', 'p-value']])

        buffer = []
        for one_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 2)):
            contingency_table = get_cont_table(one_simplex[0], one_simplex[1], bipartite_matrix)
            table_str = str(contingency_table[0, 0]) + '_' + str(contingency_table[0, 1]) + '_' + \
                        str( contingency_table[1, 0]) + '_' + str(contingency_table[1, 1])

            phi = phi_coefficient_table(contingency_table)

            chi2, p = dictionary[table_str]
            buffer.append([one_simplex[0], one_simplex[1], phi, p])
            writer = csv.writer(csvFile)
            writer.writerows(buffer)

            # empty the buffer
            buffer = []

        writer = csv.writer(csvFile)
        writer.writerows(buffer)

def find_unique_cubes(matrix, save_name):
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
    json.dump(table_set, open(save_name + "_cube_list.json", 'w'))

def get_pairwise_pvalue_lower_than_alpha(matrix, save_name, alpha=0.01):
    #TODO Obsolete
    pvaldictio = {}

    for one_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 2)):
        row_1 = matrix1.getrow(one_simplex[0]).toarray().flatten().astype(np.int32)
        row_2 = matrix1.getrow(one_simplex[1]).toarray().flatten().astype(np.int32)

        cont_cube = get_cont_table(row_1, row_2)
        expected = mle_2x2_ind(cont_cube)

        pval = chisq_test(cont_cube, expected, df=1)[1]
        if  pval < alpha:
            pvaldictio[str(one_simplex)] = pval

    json.dump(pvaldictio, open(save_name + "_pair_pvalues_lower_than_alpha_dictio.json", 'w'))

def get_triplets_pvalue_lower_than_alpha(matrix, save_name, alpha=0.01):
    # TODO Obsolete
    pvaldictio = {}

    for two_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 3)):

        cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix)
        expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)

        if expected is not None:
            pval = chisq_test(cont_cube, expected, df=1)[1]
            if  pval < alpha:
                pvaldictio[str(two_simplex)] = pval

    json.dump(pvaldictio, open(save_name + "_triplet_pvalues_lower_than_alpha_dictio.json", 'w'))


def count_impossible_triplets(matrix):
    """
    Count the number of invalid contingency cubes.
    TODO : this function could be moved
    :param matrix: (np.array) Presence/absence matrix
    :return: (int) number of invalid contingency cubes in the presence/absence matrix.
    """
    count = 0
    for two_simplex in tqdm(itertools.combinations(range(matrix.shape[0]), 3)):

        cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], matrix)
        count += find_if_invalid_cube(cont_cube)

    print('Number of invalid cubes : ', count)
    return count

@jit(nopython=True)
def find_if_invalid_cube(cont_cube):
    """
    Function used to know whether a sufficient configuration contains a zero (which indicates an invalid table).
    :param cont_cube: (np.array of ints) 2X2X2 contingency cube.
    :return: 1 if the table is invalid, 0 otherwise.
    """
    xij_ = np.sum(cont_cube, axis=0)
    if np.count_nonzero(xij_) != 4:
        return 1
    xi_k = np.sum(cont_cube, axis=2)
    if np.count_nonzero(xi_k) != 4:
        return 1
    x_jk = np.sum(cont_cube, axis=1).T
    if np.count_nonzero(x_jk) != 4:
        return 1

    return 0

def pvalues_for_cubes(file_name):

    """
    Find the p-values for the unique contingency cubes found with the function find_unique_cubes
    :param file_name: (str) Path to the file obtained with the function find_unique_cubes. This also acts as a
                            savename for the dictionary table : (chi^2 statistics, p-value). The keys of the dictionary
                            are actually strings where we flattened the 2X2X2 contingency cubes and separate each entry
                            by underscores.
    :return: None
    """

    with open(file_name + '_cube_list.json') as json_file:
        table_set = json.load(json_file)

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

            else :
                pvaldictio[table_id] = str(expected)

        json.dump(pvaldictio, open(data_name + "_asymptotic_cube_pval_dictio.json", 'w'))


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
    with open(savename +'.csv', 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])


        for two_simplex in tqdm(itertools.combinations(range(bipartite_matrix.shape[0]), 3)):
            cont_cube = get_cont_cube(two_simplex[0], two_simplex[1], two_simplex[2], bipartite_matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))

            try :
                chi2, p = dictionary[table_str]
            except:
                #TODO Change for None?
                chi2, p = 0.0, 1.0

            writer.writerow([two_simplex[0], two_simplex[1], two_simplex[2], p])

def significant_triplet_from_csv(csvfilename, alpha, savename):
    """
    Extract the significant triplets (triplets with p-value < alpha)
    :param csvfilename: (str) Path to the file obtained with the function save_triplets_p_values_dictionary.
    :param alpha: (float) Threshold of significance
    :param savename: (str) Name of the csv file where we save information
    :return: None
    """

    with open(csvfilename, 'r') as csvfile, open(savename + '.csv', 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            try :
                p = float(row[-1])
                if p < alpha:
                    writer = csv.writer(fout)
                    writer.writerow([int(row[0]), int(row[1]), int(row[2]),  p])
            except:
                pass

def triangles_p_values_tuple_dictionary(csvfile, savename, dictionary, matrix):
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


    with open(csvfile, 'r') as csvfile, open(savename, 'w',  newline='') as fout:
        reader = csv.reader(csvfile)
        writer = csv.writer(fout)
        writer.writerows([['node index 1', 'node index 2', 'node index 3', 'p-value']])
        next(reader)
        for row in tqdm(reader):
            row = row[:3]
            row = [int(i) for i in row]

            row.sort()
            row = tuple(row)
            cont_cube = get_cont_cube(row[0], row[1], row[2], matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))
            try :
                chi2, p = dictionary[table_str]
                writer.writerow([row[0], row[1], row[2], p])
            except:
                pass

if __name__ == '__main__':


    # Options to decide if we use the step method (recommended) or the systematic method (longer and does not create
    # a simplicial complex. Use step_method = False for this one)
    step_method = False

    # Choose the name of the directory (dirName) where to save the files and the 'prefix' name of each created files
    # (data_name)
    dirName = 'Directory'
    data_name = 'Data'

    # Choose the alpha parameter to use throughout the analysis.
    alpha = 0.01

    # Enter the path to the presence/absence matrix :
    matrix1 = np.load(r'PATH_TO_MATRIX')
    matrix1 = matrix1.astype(np.int64)

    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    data_name = os.path.join(dirName, data_name)


    ########## First step : Extract all the unique tables

    print('Step 1 : Extract all the unique tables')

    # Finds all unique tables
    find_unique_tables(matrix1, data_name)

    ######### Second step : Extract all the pvalues with an asymptotic distribution

    print('Step 2: Extract pvalues for all tables with an asymptotic distribution')

    pvalues_for_tables(data_name)

    ######### Third step : Find table for all links and their associated pvalue

    print('Step 3 : Find table for all links and their associated pvalue')

    with open(data_name + '_asymptotic_pval_dictio.json') as jsonfile:
        dictio = json.load(jsonfile)

        save_pairwise_p_values_phi_dictionary(matrix1, dictio, data_name + '_asymptotic_pvalues')


    ######### Fourth step : Choose alpha and extract the network

    print('Step 4 : Generate network and extract edge_list for a given alpha')

    g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

    nx.write_edgelist(g, data_name + '_asymptotic_edge_list_' + str(alpha)[2:] + '.txt', data=True)

    print('Number of nodes : ', g.number_of_nodes())
    print('Number of links : ', g.number_of_edges())

    ######### Fifth step : Extract all the unique cubes

    print('Step 5 : Extract all the unique valid cubes')

    find_unique_cubes(matrix1, data_name)

    ######### Sixth step : Extract pvalues for all cubes with an asymptotic distribution

    print('Step 6: Extract pvalues for all cubes with an asymptotic distribution')

    pvalues_for_cubes(data_name)

    ######## Seventh step : Find cube for all triplets and their associated pvalue

    if not step_method:

        print('Step 7 : Find cube for all triplets and their associated pvalue')

        with open(data_name + "_asymptotic_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)

            save_triplets_p_values_dictionary(matrix1, dictio, data_name + '_asymptotic_cube_pvalues')

        significant_triplet_from_csv(data_name + '_asymptotic_cube_pvalues.csv', alpha, data_name + '_asymptotic_hyperlinks_' + str(alpha)[2:])

        exit()

    else:
        print('Step method : ')


    ######## Fifth step : Find all triangles in the previous network

        print('Step 7 : Finding all empty triangles in the network')

        g = read_pairwise_p_values(data_name + '_asymptotic_pvalues.csv', alpha)

        save_all_triangles(g, data_name + '_asymptotic_triangles_' + str(alpha)[2:])

        print('Number of triangles : ', count_triangles_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv'))

    ######## Sixth step : Find all the p-values for the triangles under the hypothesis of homogeneity

        print('Step 8 : Find all the p-values for the triangles under the hypothesis of homogeneity')

        with open(data_name + "_asymptotic_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)


            triangles_p_values_tuple_dictionary(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '.csv', data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', dictio, matrix1)

    ######## Fifth step : Exctract all 2-simplices

        print('Extract 2-simplices')

        significant_triplet_from_csv(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_asymptotic_2-simplices_' + str(alpha)[2:])

    exit()

    # THIS ONE GIVES ALL TRIANGLES THAT CONVERGED REGARDLESS OF THEIR P-VALUE (NO ALPHA NEEDED)
    #extract_converged_triangles(data_name + '_asymptotic_triangles_' + str(alpha)[2:] + '_pvalues.csv', data_name + '_converged_triangles')


    ################# DONE ###################