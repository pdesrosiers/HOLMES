from holmes.generative_model.synth_analyser import *


with open(r'sc_20_nodes\data_none_fg.pkl', 'rb') as fg_file:
    factorgraph = pickle.load(fg_file)

effective_facet_list_dictio = factorgraph.effective_facet_list

effective_one_simps = effective_facet_list_dictio[2]
effective_two_simps = effective_facet_list_dictio[3]

# Choose the alpha parameter to use throughout the analysis.
alpha = 0.01

dictionary_of_truth = {'vp_one': [], 'fn_one': [], 'fp_one': [], 'vp_two': [], 'fn_two': [], 'fp_two': []}
directory = r'sc_20_nodes\100'

for i in range(100):
    # Choose the name of the directory (dirName) where to save the files and the 'prefix' name of each created files
    # (data_name)
    # dirName = 'Directory'
    # data_name = 'Data'

    # Enter the path to the presence/absence matrix :
    matrix1 = np.load(directory + r'\data_none_bipartite_' + str(i) + '.npy')
    matrix1 = matrix1.astype(np.int64)

    ## Create target Directory if don't exist
    # if not os.path.exists(dirName):
    #    os.mkdir(dirName)
    #    print("Directory ", dirName, " Created ")
    # else:
    #    print("Directory ", dirName, " already exists")

    # data_name = os.path.join(dirName, data_name)

    ########## First step : Extract all the unique tables

    print('Step 1 : Extract all the unique tables')

    # Finds all unique tables
    all_unique_tables = find_unique_tables(matrix1)

    ######### Second step : Extract all the pvalues with an asymptotic distribution

    print('Step 2: Extract pvalues for all tables with an asymptotic distribution')

    pvaldictio_for_tables = pvalues_for_tables(all_unique_tables)

    ######### Third step : Find table for all links and their associated pvalue

    print('Step 3 : Find table for all links and their associated pvalue')

    # with open(data_name + '_asymptotic_pval_dictio.json') as jsonfile:
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

with open(directory + r'\dictionary_of_truth.pkl', 'wb') as f:
    pickle.dump(dictionary_of_truth, f)