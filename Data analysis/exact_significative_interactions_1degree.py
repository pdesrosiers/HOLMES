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
    :param cont_tab: (np.array of ints) contingency table, can be in any dimension
    :param expected: (np.array of ints) contingency table of the expected values under a given log-linear model
    :return: The chi^2 test statistics (float between 0 and inf)
    """
    teststat = 0
    cont_tab = cont_tab.flatten()
    expected = expected.flatten()

    for i in range(len(cont_tab)):
        teststat += (cont_tab[i] - expected[i]) ** 2 / expected[i]

    return teststat

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    """
    Computes the test statistics (using test_statistics) and its p-value using an array of test statistics.
    :param cont_table: (np.array of ints) contingency table, can be in any dimension
    :param expected_table: (np.array of ints) contingency table of the expected values under a given log-linear model
    :param sampled_array: (np.array of floats) statistics generated using multinomial distributions, a specific log-
                          linear model and the function build_chisqlist.
    :return: The chi^2 statistics computed between cont_table and expected_table and its exact p-value.
    """
    if np.any(expected_table == 0):
        test_stat = 0
        pval = 1
    else:
        test_stat = test_statistics(cont_table, expected_table)
        cdf = np.sum((np.array(sampled_array) < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval


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
    TODO : Function is not used in this script
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


def pvalues_for_tables(file_name, nb_samples, N):
    """
    Find the p-values for the unique contingency tables found with the function find_unique_tables
    :param file_name: (str) Path to the file obtained with the function find_unique_tables. This also acts as a
                            savename for the dictionary table : (chi^2 statistics, p-value). The keys of the dictionary
                            are actually strings where we flattened the 2X2 contingency table and separate each entry
                            by underscores.
    :return: None
    TODO : To distinguish cases where we can't compute the pvalue from cases where the pvalue is actually 0, we should
    change the first conditional block (if not find_if_invalid_table(table):)
                else:
                pvaldictio[table_id] = (0.0, 1.0)
    for something like :
                 else:
                pvaldictio[table_id] = (0.0, None)

    and check every other step of the process so that this 'None' entry in the CSV file can be ignored by the functions
    """

    with open(file_name + "_table_list.json") as json_file:
        table_set = json.load(json_file)

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        # Max index used in range() :
        for it in tqdm(range(len(table_set))):
            table_id = table_set[it]
            table = np.random.rand(2, 2)
            table_id_list = str.split(table_id, '_')
            # print(table_id, table_id_list)
            table[0, 0] = int(table_id_list[0])
            table[0, 1] = int(table_id_list[1])
            table[1, 0] = int(table_id_list[2])
            table[1, 1] = int(table_id_list[3])

            if not find_if_invalid_table(table):
                expected_original = mle_2x2_ind(table)
                problist = mle_multinomial_from_table(expected_original, N)
                samples = multinomial_problist_cont_table(N, problist, nb_samples)
                chisqlist = build_chisqlist(samples, nb_samples)

                if len(chisqlist) == nb_samples:

                    pvaldictio[table_id] = sampled_chisq_test(table, expected_original, chisqlist)
                else:
                    pvaldictio[table_id] = (0.0, 1.0)
            else:
                pvaldictio[table_id] = (0.0, 1.0)

        json.dump(pvaldictio, open(file_name + "_exact_1deg_pval_dictio.json", 'w'))

@jit(nopython=True)
def build_chisqlist(samples, nb_samples):
    """
    Computes the chi^2 statistic between sampled contingency tables (from a multinomial distribution) and the expected
    tables under the model of independence. The result is a list of chi^2 statistics.
    :param samples: (np.array of contingency tables)
    :param nb_samples: (int) number of samples in samples (TODO could be computed within the function)
    :return: list of floats that represent chi^2 statistics
    """
    chisqlist = []
    for i in range(nb_samples):
        sample = samples[i, :, :]
        expected = mle_2x2_ind(sample)
        if np.any(expected == 0):
            break
        else:
            chisqlist.append(test_statistics(sample, expected))
    return chisqlist

@jit(nopython=True)
def build_chisqlist_cube(samples, nb_samples):
    """
    Computes the chi^2 statistic between sampled contingency cubes (from a multinomial distribution) and the expected
    cubes under the model of no second order interaction. The result is a list of chi^2 statistics.
    :param samples: (np.array of contingency cubes)
    :param nb_samples: (int) number of samples in samples (TODO could be computed within the function)
    :return: list of floats that represent chi^2 statistics
    TODO : We can probably remove the part with :
                if expected is not None:
                for entry in expected.flatten():
                    if entry < 0.01:
                        switch = True
                        break
            if switch:
                break
    I think that iterative_proportional_fitting_AB_AC_BC_no_zeros already checks if entries in the contingency cube
    are too close to zero (meaning that the sampled table and the expected table are the same, because the MLE does not
    exist).
    """
    chisqlist = []
    for i in range(nb_samples):
        switch = False
        sample = samples[i, :, :, :]
        if not find_if_invalid_cube(sample):
            expected = iterative_proportional_fitting_AB_AC_BC_no_zeros(sample)
            if expected is not None:
                for entry in expected.flatten():
                    if entry < 0.01:
                        switch = True
                        break
            if switch:
                break
            else:
                chisqlist.append(test_statistics(sample, expected))
        else:
            break
    return chisqlist

@jit(nopython=True)
def find_if_invalid_table(cont_table):
    """
    Find if there are zero entries in the sufficient configurations of the table. If so, the table is invalid for
    computation of the p-value (MLE does not exist)
    :param cont_table: (np.array) a 2X2 contingency table
    :return: 1 if table is invalide, 0 otherwise.
    """
    x_j = np.sum(cont_table, axis=0)
    if np.count_nonzero(x_j) != 2:
        return 1
    xi_ = np.sum(cont_table, axis=1)
    if np.count_nonzero(xi_) != 2:
        return 1

    return 0

@jit(nopython=True)
def find_if_invalid_cube(cont_cube):
    """
    Find if there are zero entries in the sufficient configurations of the table. If so, the table is invalid for
    computation of the p-value (MLE does not exist)

    :param cont_cube: (np.array) a 2X2X2 contingency table
    :return: 1 if table is invalide, 0 otherwise.
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
        writer.writerows([['node index 1', 'node index 2','phi-coefficient', 'p-value']])

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

        table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))
        if table_str not in table_set:
            table_set.add(table_str)

    table_set = list(table_set)
    print('How many different cubes : ', len(table_set))
    json.dump(table_set, open(save_name + "_cube_list.json", 'w'))

def pvalues_for_cubes(file_name, nb_samples, N):
    """
    Find the p-values for the unique contingency cubes found with the function find_unique_cubes
    :param file_name: (str) Path to the file obtained with the function find_unique_cubes. This also acts as a
                            savename for the dictionary table : (chi^2 statistics, p-value). The keys of the dictionary
                            are actually strings where we flattened the 2X2X2 contingency cubes and separate each entry
                            by underscores.
    :param nb_samples: (int) Number of samples that we want to generate the exact distribution
    :param N:  (int) Number of observations we want in each sampled contingency tables.
    :return: None
    """

    with open(file_name + '_cube_list.json') as json_file:
        table_set = list(json.load(json_file))

        #### From the different tables : generate the chisqdist :

        pvaldictio = {}

        for it in tqdm(range(len(table_set))):
            switch = False
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



            if not find_if_invalid_cube(table):
                expected_original = iterative_proportional_fitting_AB_AC_BC_no_zeros(table)
                for entry in expected_original.flatten():
                    if entry < 0.001:
                        switch = True
                if switch:
                    continue
                    #pvaldictio[table_id] = (0.0, 1.0)
                elif expected_original is not None:

                    problist = mle_multinomial_from_table(expected_original, N)
                    samples = multinomial_problist_cont_cube(N, problist, nb_samples)
                    
                    chisqlist = build_chisqlist_cube(samples, nb_samples)

                    if len(chisqlist) == nb_samples:

                        pvaldictio[table_id] = sampled_chisq_test(table, expected_original, chisqlist)
                        print('FOUNDONE', table_id)
                    else:
                        continue
                        #pvaldictio[table_id] = (0.0, 1.0)
            else:
                continue
                #pvaldictio[table_id] = (0.0, 1.0)

        json.dump(pvaldictio, open(data_name + "_exact_1deg_cube_pval_dictio.json", 'w'))


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
                p = dictionary[table_str]

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

def build_simplices_list(matrix, two_simplices_file, one_simplices_file, alpha):
    """
    Build a CSV file containing all zero, one and two simplices that are significant. To obtain a facet list, we would
    need to remove the included ' facets ' using the '' prune '' function in the SCM package.
    (See https://github.com/jg-you/scm for the prune function)
    :param matrix: (np.array) Presence/absence matrix
    :param two_simplices_file: CSV file where either candidates for hyperlinks or for 2-simplices are found
    :param one_simplices_file: CSV file where candidates for links (1-simplices) are found
    :param alpha:   (float) Threshold of significance
    :return:
    """
    #open('facet_list.txt', 'a').close()

    with open('facet_list.txt', 'w') as facetlist:

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
    return

def triangles_p_values_AB_AC_BC_dictionary(csvfile, savename, dictionary, matrix):

    """
    Fetches the p-values of triplets that form a triangle after the first step of the method.
    :param csvfile: (str) Path to the file obtained with the function save_all_triangles.
    :param savename: (str) Name of the csv file where we save information
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

            cont_cube = get_cont_cube(int(row[0]), int(row[1]), int(row[2]), matrix)

            table_str = str(int(cont_cube[0, 0, 0])) + '_' + str(int(cont_cube[0, 0, 1])) + '_' + str(
                int(cont_cube[0, 1, 0])) + '_' + str(int(cont_cube[0, 1, 1])) + '_' + str(
                int(cont_cube[1, 0, 0])) + '_' + str(int(cont_cube[1, 0, 1])) + '_' + str(
                int(cont_cube[1, 1, 0])) + '_' + str(int(cont_cube[1, 1, 1]))

            try :

                chi2, p = dictionary[table_str]

            except:

                p = dictionary[table_str]

            writer.writerow([row[0], row[1], row[2], p])


if __name__ == '__main__':
    # Options to decide if we use the step method (recommended) or the systematic method (longer and does not create
    # a simplicial complex. Use step_method = False for this one)
    step_method = True

    # Choose the name of the directory (dirName) where to save the files and the 'prefix' name of each created files
    # (data_name)
    dirName = 'New_directory'
    data_name = 'Data'

    # Choose the alpha parameter to use throughout the analysis.
    alpha = 0.01

    # Number of samples that will generate our exact distribution (higher is better, but more time consuming)
    nb_samples = 1000000

    # Enter the path to the presence/absence matrix :
    matrix1 = np.load(r'PATH')
    matrix1 = matrix1.astype(np.int64)

    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    else:
        print("Directory ", dirName, " already exists")

    data_name = os.path.join(dirName, data_name)


    ###### First step : Extract all the unique tables

    print('Step 1 : Extract all the unique tables')

    # Finds all unique tables
    find_unique_tables(matrix1, data_name)

    ######## Second step : Extract pvalues for all tables with an exact Chi1 distribution

    print('Step 2: Extract pvalues for all tables with an exact Chi1 distribution')

    pvalues_for_tables(data_name, nb_samples, matrix1.shape[1])

    ######## Third step : Find table for all links and their associated pvalue

    print('Step 3 : Find table for all links and their associated pvalue')

    with open(data_name + '_exact_pval_dictio.json') as jsonfile:
        dictio = json.load(jsonfile)

        save_pairwise_p_values_phi_dictionary(matrix1, dictio, data_name + '_exact_pvalues')


    ######## Fourth step : extract the network for a given alpha

    print('Step 4 : Generate network and extract edge_list for a given alpha')

    g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)
    nx.write_edgelist(g, data_name + '_exact_edge_list_' + str(alpha)[2:] + '.txt', data=True)

    print('Number of nodes : ', g.number_of_nodes())
    print('Number of links : ', g.number_of_edges())

    ######## Fifth step : Extract all the unique cubes

    print('Step 5 : Extract all the unique cubes')

    find_unique_cubes(matrix1, data_name)

    ###### Sixth step : Extract pvalues for all cubes with an exact CHI 1 distribution

    print('Step 6: Extract pvalues for all tables with an exact CHI 1 distribution')

    pvalues_for_cubes(data_name, nb_samples, matrix1.shape[1])

    ######## Seventh step : Find cube for all triplets and their associated pvalue

    if not step_method:

        print('Step 7 : Find cube for all triplets and their associated pvalue')

        with open(data_name + "_exact_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)

            save_triplets_p_values_dictionary(matrix1, dictio, data_name + '_exact_cube_pvalues')

        significant_triplet_from_csv(data_name + '_exact_cube_pvalues.csv', alpha, data_name + '_exact_hyperlinks_'  + str(alpha)[2:])

        exit()

    else:
        print('Step Method : ')

    ######## Fifth step : Find all triangles in the previous network

        print('Finding all empty triangles in the network')

        g = read_pairwise_p_values(data_name + '_exact_pvalues.csv', alpha)

        save_all_triangles(g, data_name + '_exact_triangles_' + str(alpha)[2:])

        print('Number of triangles : ', count_triangles_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv'))

    ######## Sixth step : Find all the p-values for the triangles under the hypothesis of homogeneity

        print('Find all the p-values for the triangles under the hypothesis of homogeneity')

        with open(data_name + "_exact_cube_pval_dictio.json") as jsonfile:
            dictio = json.load(jsonfile)

            triangles_p_values_AB_AC_BC_dictionary(data_name + '_exact_triangles_' + str(alpha)[2:] + '.csv', data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', dictio, matrix1)

    ######## Fifth step : Extract all 2-simplices

        print('Extract 2-simplices')

        significant_triplet_from_csv(data_name + '_exact_triangles_' + str(alpha)[2:] + '_pvalues.csv', alpha, data_name + '_exact_2-simplices_' + str(alpha)[2:])

    exit()

    ################# DONE ###################
