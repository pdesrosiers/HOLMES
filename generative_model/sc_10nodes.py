from base import *
from loglin_model import *
from metropolis_sampler import *
from script_sampler import *
from another_sign_int import *
from matplotlib.ticker import MultipleLocator
from matplotlib import rc

def problist_to_table_old(prob_dist, sample_size):

    dimension = len(prob_dist) - 1
    table = np.random.rand(dimension)
    reshape = np.log(dimension)/np.log(2)
    table = np.reshape(table, np.repeat(2, reshape))

    for key in list(prob_dist.keys()):

        if key != 'T':

            table[key] = prob_dist[key]

    #table = np.roll(table, (1, 1), (0, 1))

    return table * sample_size

def get_cont_table(u_idx, v_idx, matrix):
    # Computes the 2X2 contingency table for the occurrence matrix
    row_u_present = matrix[u_idx, :]
    row_v_present = matrix[v_idx, :]

    row_u_not_present = 1 - row_u_present
    row_v_not_present = 1 - row_v_present

    # u present, v present
    table00 = np.dot(row_u_present, row_v_present)

    # u present, v NOT present
    table01 = np.dot(row_u_present, row_v_not_present)

    # u NOT present, v present
    table10 = np.dot(row_u_not_present, row_v_present)

    # u NOT present, v NOT present
    table11 = np.dot(row_u_not_present, row_v_not_present)

    return np.array([[table00, table01], [table10, table11]])

def problist_to_2x2_table(prob_dist, idx1, idx2, sample_size):

    table = np.random.rand(2,2)
    p_00 = 0
    p_10 = 0
    p_01 = 0
    p_11 = 0
    for key in list(prob_dist.keys()):
        if key[idx1] == 0 and key[idx2] == 0:
            p_00 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0:
            p_10 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1:
            p_01 += prob_dist[key]
        else:
            p_11 += prob_dist[key]

    table[0, 0] = p_00
    table[1, 0] = p_10
    table[0, 1] = p_01
    table[1, 1] = p_11

    return table * sample_size

def problist_to_2x2x2_cube(prob_dist, idx1, idx2, idx3, sample_size):

    table = np.random.rand(2,2,2)
    p_000 = 0
    p_010 = 0
    p_001 = 0
    p_011 = 0
    p_100 = 0
    p_110 = 0
    p_101 = 0
    p_111 = 0
    for key in list(prob_dist.keys()):
        if key[idx1] == 0 and key[idx2] == 0 and key[idx3] == 0 :
            p_000 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0 and key[idx3] == 0 :
            p_010 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1 and key[idx3] == 0 :
            p_001 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 1 and key[idx3] == 0:
            p_011 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 0 and key[idx3] == 1:
            p_100 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0 and key[idx3] == 1:
            p_110 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1 and key[idx3] == 1:
            p_101 += prob_dist[key]
        else:
            p_111 += prob_dist[key]

    table[0, 0, 0] = p_000
    table[0, 1, 0] = p_010
    table[0, 0, 1] = p_001
    table[0, 1, 1] = p_011
    table[1, 0, 0] = p_100
    table[1, 1, 0] = p_110
    table[1, 0, 1] = p_101
    table[1, 1, 1] = p_111

    return table * sample_size

def mutual_information(prob_dist, idx1, idx2):

    table = np.random.rand(2, 2)

    p_00 = 0
    p_10 = 0
    p_01 = 0
    p_11 = 0
    for key in list(prob_dist.keys()):
        if key[idx1] == 0 and key[idx2] == 0:
            p_00 += prob_dist[key]
        elif key[idx1] == 1 and key[idx2] == 0:
            p_10 += prob_dist[key]
        elif key[idx1] == 0 and key[idx2] == 1:
            p_01 += prob_dist[key]
        else:
            p_11 += prob_dist[key]

    table[0, 0] = p_00
    table[1, 0] = p_10
    table[0, 1] = p_01
    table[1, 1] = p_11

    py = np.sum(table, axis = 0).reshape(1, 2)
    px = np.sum(table, axis = 1).reshape(2, 1)

    pxpy = np.matmult(px, py)

    return np.sum(table * np.log(table/pxpy))


def distance_to_original(bam, original):

    sampled_table = get_cont_cube(1, 2, 0, bam)



    return np.sum(np.abs((sampled_table - original)))

    #return np.nan_to_num(np.sum((sampled_table - original/N_original*N)**2/(original/N_original*N)))

def mle_multinomial_from_table(cont_table):
    n = np.sum(cont_table)
    p_list = []
    for element in cont_table.flatten():
        p_list.append(element/n)

    return p_list

def multinomial_problist_cont_cube(nb_trials, prob_list, s=1):
    return np.random.multinomial(nb_trials, prob_list, s).reshape(s, 2, 2, 2)

def sampled_chisq_test(cont_table, expected_table, sampled_array):
    if float(0) in expected_table:
        test_stat = 0
        pval = 1
    else:
        test_stat = np.sum((cont_table - expected_table) ** 2 / expected_table)
        cdf = np.sum((sampled_array < test_stat) * 1) / len(sampled_array)
        pval = 1 - cdf
    return test_stat, pval

def chisq_formula_vector_for_cubes(cont_tables, expected):
    # Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    # via MLE or iterative proportional fitting.

    return np.nan_to_num(np.sum(np.sum(np.sum((cont_tables - expected) ** 2 / expected, axis = 1), axis = 1), axis=1))

def chisq_test_here(cont_tab, expected, df=1):
    #Computes the chisquare statistics and its p-value for a contingency table and the expected values obtained
    #via MLE or iterative proportional fitting.
    if np.any(expected == 0):
        print('HERE')
        return 0, 1
    #df = 7
    test_stat = np.sum((cont_tab-expected)**2/expected)
    p_val = chi2.sf(test_stat, df)

    return test_stat, p_val

def get_fg_vp(fg_path, N):
    fg_1simplices_list = []
    fg_2simplices_list = []

    with open(fg_path, 'rb') as fg_file:
        factorgraph = pickle.load(fg_file)
        probdist = Prob_dist(factorgraph)

        for one_simp in itertools.combinations(factorgraph.node_list, 2):

            cont_table = problist_to_2x2_table(probdist.prob_dist, one_simp[0], one_simp[1], N)
            expected_1 = mle_2x2_ind(cont_table)
            pval = chisq_test_here(cont_table, expected_1)[1]
            #print(one_simp, pval)
            if pval < alpha:
                fg_1simplices_list.append(list(one_simp))

        for two_simp in itertools.combinations(factorgraph.node_list, 3):
            cont_cube = problist_to_2x2x2_cube(probdist.prob_dist, two_simp[0], two_simp[1], two_simp[2], N)
            # print(cont_cube)
            expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
            if expected_2 is not None:
                pval = chisq_test_here(cont_cube, expected_2)[1]
            else:
                pval = 1
            if pval < alpha:
                fg_2simplices_list.append(list(two_simp))

    return fg_1simplices_list, fg_2simplices_list

def get_fg_vp_pathless(fg, N):
    fg_1simplices_list = []
    fg_2simplices_list = []


    factorgraph = fg
    probdist = Prob_dist(factorgraph)

    for one_simp in itertools.combinations(factorgraph.node_list, 2):

        cont_table = problist_to_2x2_table(probdist.prob_dist, one_simp[0], one_simp[1], N)
        expected_1 = mle_2x2_ind(cont_table)
        pval = chisq_test_here(cont_table, expected_1)[1]
        #print(one_simp, pval)
        if pval < alpha:
            fg_1simplices_list.append(list(one_simp))

    for two_simp in itertools.combinations(factorgraph.node_list, 3):
        cont_cube = problist_to_2x2x2_cube(probdist.prob_dist, two_simp[0], two_simp[1], two_simp[2], N)
        # print(cont_cube)
        expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
        if expected_2 is not None:
            pval = chisq_test_here(cont_cube, expected_2)[1]
        else:
            pval = 1
        if pval < alpha:
            fg_2simplices_list.append(list(two_simp))

    return fg_1simplices_list, fg_2simplices_list

def get_complete_list_with_pvals(fg_path, N ):
    complete_list_with_pval = []


    with open(fg_path, 'rb') as fg_file:
        factorgraph = pickle.load(fg_file)
        probdist = Prob_dist(factorgraph)

        for one_simp in itertools.combinations(factorgraph.node_list, 2):


            cont_table = problist_to_2x2_table(probdist.prob_dist, one_simp[0], one_simp[1], N)
            if one_simp[0] == 3 and one_simp[1] == 4:
                #print(cont_table)
                pass
            expected_1 = mle_2x2_ind(cont_table)
            pval = chisq_test_here(cont_table, expected_1)[1]
            complete_list_with_pval.append((list(one_simp), pval))

        for two_simp in itertools.combinations(factorgraph.node_list, 3):
            cont_cube = problist_to_2x2x2_cube(probdist.prob_dist, two_simp[0], two_simp[1], two_simp[2], N)
            # print(cont_cube)
            if two_simp[0] == 0 and two_simp[1] == 1 and two_simp[2] == 2:
                #print(cont_cube)
                pass
            expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
            if expected_2 is not None:
                pval = chisq_test_here(cont_cube, expected_2)[1]
            else:
                pval = 1
            complete_list_with_pval.append((list(two_simp), pval))


    return complete_list_with_pval

if __name__ == '__main__':
    observations = 1000
    alpha = 0.01
    build_sc = False
    switch = True
    vp_1simp = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7], [4, 8], [7, 8], [7, 9], [8, 9]]
    vp_2simp = [[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3],  [7, 8, 9]]

    nb_it = []

    # This loop finds a factor graph without induced interactions (interactions that we didn't specify)
    # TODO Even though that's the goal of  this loop, I think there are missing steps. For instance, we should also
    # check if all interactions in the factor graph are in vp_1simp and vp_2simp
    i = 1
    switch = True
    while switch:

        print(i)
        factorgraph = FactorGraph([[3, 4], [3, 5], [4, 5], [5, 6], [5, 7], [6, 7], [4, 8],[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3],  [7, 8, 9]], N=observations, alpha=alpha, build_sc=build_sc)

        fg_1simplices_list, fg_2simplices_list = get_fg_vp_pathless(factorgraph, observations)

        #if len(fg_1simplices_list) == 16 and len(fg_2simplices_list) == 5:
        bad1simp = False
        bad2simp = False
        for onesimp in vp_1simp:
            if onesimp not in fg_1simplices_list:
                bad1simp = True
                break

        for twosimp in vp_2simp:
            if twosimp not in fg_2simplices_list:
                bad2simp = True
                break

        if not bad1simp and not bad2simp:

            switch = False
            #print(fg_1simplices_list)
            #print(fg_2simplices_list)
        i += 1
    print(fg_1simplices_list)
    print(fg_2simplices_list)


    with open(r'Directory\fg.pkl', 'wb') as output:
        pickle.dump(factorgraph, output, pickle.HIGHEST_PROTOCOL)

    with open(r'Directory\fg.pkl', 'rb') as fg_file:
        factorgraph = pickle.load(fg_file)
        probdist = Prob_dist(factorgraph)

    #This loop generates k Biadjacency matrices AKA presence/absence matrices.
    #The variable observations gives the number of sites we want to generate for each matrix.

    k = 10

    for i in np.arange(0, k, 1):
        state = np.random.randint(2, size=len(factorgraph.node_list))
        print(state)

        energy_obj = Energy(state, factorgraph)
        proposer = BitFlipProposer(factorgraph, energy_obj, state)
        sampler = Sampler(proposer, temperature=1, initial_burn=5, sample_burn=2**len(factorgraph.node_list))
        sampler.sample(observations)
        #print(sampler.results['nb_rejected'])
        #print(sampler.results['nb_success'])

        bipartite = build_bipartite(sampler.results['sample'])
        np.save(r'Directory\bipartite_' + str(i), bipartite)

    exit()

    #### TODO This stuff is irrelevant for now.

    fg_1simplices_list, fg_2simplices_list = get_fg_vp('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/data_simplicial_10nodes/fg.pkl', observations)

    with open('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/data_simplicial_10nodes/fg.pkl', 'rb') as fg_file:
        factorgraph = pickle.load(fg_file)
        probdist = Prob_dist(factorgraph)
    print(fg_1simplices_list)
    print(fg_2simplices_list)
    for tupl in get_complete_list_with_pvals('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/data_simplicial_10nodes/fg.pkl', observations):
        print(tupl)
    #print(get_complete_list_with_pvals('/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/data_simplicial_10nodes/fg.pkl', observations))
    #exit()
    path_to_bams = '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/data_simplicial_10nodes/bipartite_'
    list_of_pval_list = []
    pval_list = []
    distance_list = []
    link_count = []
    alpha = 0.01


    pval_list_vp_link = []
    pval_list_fp_link = []
    pval_list_vp_2simp = []
    pval_list_fp_2simp = []
    nb_found_links_list = []
    nb_found_twosimp_list = []
    mean = np.array([[0, 0], [0, 0]], dtype=np.float64)
    total_nb_vp_link_list = []
    total_nb_fp_link_list = []
    total_nb_vp_2simp_list = []
    total_nb_fp_2simp_list = []
    #mean = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.float64)
    for i in np.arange(0, 100, 1):
        link_found_list = []
        twosimp_found_list = []
        print(i)
        list_of_1simp_found = []

        bam = np.load(path_to_bams + str(i) + '.npy')
        nb_found_links = 0
        for one_simp in itertools.combinations(factorgraph.node_list, 2):

            cont_table = get_cont_table(one_simp[0], one_simp[1], bam)

            #if one_simp[0] == 3 and one_simp[1] == 4:
            #    mean += cont_table
            expected_1 = mle_2x2_ind(cont_table)
            pval = chisq_test_here(cont_table, expected_1)[1]
            if list(one_simp) in fg_1simplices_list:
                pval_list_vp_link.append(pval)
                if pval < alpha:
                    link_found_list.append(1)
                else:
                    #print(one_simp, pval)
                    link_found_list.append(0)
            else:
                if pval < alpha:
                    pval_list_fp_link.append(pval)
                    print('FP LINK : ', one_simp)
            if pval < alpha:
                list_of_1simp_found.append(list(one_simp))
                #print(one_simp, pval)
                nb_found_links += 1
        total_nb_vp_link_list.append(np.sum(link_found_list))
        total_nb_fp_link_list.append(nb_found_links - np.sum(link_found_list))
        #print(list_of_1simp_found)
        #print(total_nb_vp_link_list)
        #print(total_nb_fp_link_list)
        nb_found_twosimp = 0
        #print(list_of_1simp_found)
        if build_sc:
            for comb_of_3 in itertools.combinations(list_of_1simp_found, 3):

                nodeset = set()

                for ls in comb_of_3:
                    for node in ls:
                        nodeset.add(node)
                if len(nodeset) == 3:
                    #print(nodeset)
                    nodelist = list(nodeset)
                    nodelist.sort()
                    #print(nodelist)
                    cont_cube = get_cont_cube(nodelist[0], nodelist[1], nodelist[2], bam)
                    # print(cont_cube)
                    expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
                    if expected_2 is not None:
                        pval = chisq_test_here(cont_cube, expected_2)[1]
                    else:
                        pval = 1
                    #print(nodelist, fg_2simplices_list)
                    if nodelist in fg_2simplices_list:
                        pval_list_vp_2simp.append(pval)
                        if pval < alpha:
                            twosimp_found_list.append(1)
                        else:
                            twosimp_found_list.append(0)
                    else:
                        if pval < alpha:
                            pval_list_fp_2simp.append(pval)
                    if pval < alpha:
                        #print(nodelist, pval)
                        nb_found_twosimp += 1
            nb_found_twosimp_list.append(nb_found_twosimp)
            total_nb_vp_2simp_list.append(np.sum(twosimp_found_list))
            total_nb_fp_2simp_list.append(nb_found_twosimp - np.sum(twosimp_found_list))

        else:
            for two_simp in itertools.combinations(factorgraph.node_list, 3):
                cont_cube = get_cont_cube(two_simp[0], two_simp[1], two_simp[2], bam)
                if two_simp[0] == 0 and two_simp[1] == 1 and two_simp[2] == 2:
                    #mean += cont_cube
                    pass
                    #print(cont_cube)
                expected_2 = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube)
                if expected_2 is not None:
                    pval = chisq_test_here(cont_cube, expected_2)[1]
                else :
                    pval = 1
                if list(two_simp) in fg_2simplices_list:
                    pval_list_vp_2simp.append(pval)
                    if pval < alpha:
                        twosimp_found_list.append(1)
                    else:
                        twosimp_found_list.append(0)
                else:
                    if pval < alpha:
                        pval_list_fp_2simp.append(pval)
                if pval < alpha:
                    #print(two_simp, pval)
                    nb_found_twosimp += 1

            nb_found_twosimp_list.append(nb_found_twosimp)
            total_nb_vp_2simp_list.append(np.sum(twosimp_found_list))
            total_nb_fp_2simp_list.append(nb_found_twosimp - np.sum(twosimp_found_list))

            #nb_found_twosimp_list.append(nb_found_twosimp)

    print(total_nb_vp_link_list)

    rc('text', usetex=True)
    rc('font', size=16)
    fig1, ax1 = plt.subplots()
    n, b, p = plt.hist(total_nb_vp_link_list, bins=np.arange(0, 50, 1) - 0.5, color='#00a1ffff', rwidth=0.8)
    plt.plot([np.mean(total_nb_vp_link_list), np.mean(total_nb_vp_link_list)], [0, np.max(n)], 'k--',
             label='Valeur moyenne')
    plt.xlabel(r'Nombre de vrais $1$-simplexes (total = 16)')
    plt.ylabel(r'Nombre de r\'ealisations ')
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    plt.legend(loc=0)
    #plt.xlim(0, 8)

    print(total_nb_fp_link_list)

    fig2, ax2 = plt.subplots()
    n, b, p = plt.hist(total_nb_fp_link_list, bins=np.arange(0, 50, 1) - 0.5, color='#00a1ffff', rwidth=0.8)
    plt.plot([np.mean(total_nb_fp_link_list), np.mean(total_nb_fp_link_list)], [0, np.max(n)], 'k--',
             label='Valeur moyenne')
    plt.xlabel(r'Nombre de faux $1$-simplexes')
    plt.ylabel(r'Nombre de r\'ealisations ')
    #plt.xlim(10, 30)
    plt.legend(loc=0)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    print(total_nb_vp_2simp_list)

    fig3, ax3 = plt.subplots()
    n, b, p = plt.hist(total_nb_vp_2simp_list, bins=np.arange(0, 50, 1) - 0.5, color='#41d936ff', rwidth=0.8)
    plt.plot([np.mean(total_nb_vp_2simp_list), np.mean(total_nb_vp_2simp_list)], [0, np.max(n)], 'k--',
             label='Valeur moyenne')
    plt.xlabel(r'Nombre de vrais $2$-simplexes (total = 5)')
    plt.ylabel(r'Nombre de r\'ealisations ')
    #plt.xlim(0, 6)
    plt.legend(loc=0)
    ax3.xaxis.set_minor_locator(MultipleLocator(1))

    print(total_nb_fp_2simp_list)
    fig4, ax4 = plt.subplots()
    n, b, p = plt.hist(total_nb_fp_2simp_list, bins=np.arange(0, 50, 1) - 0.5, color='#41d936ff', rwidth=0.8)
    plt.plot([np.mean(total_nb_fp_2simp_list), np.mean(total_nb_fp_2simp_list)], [0, np.max(n)], 'k--',
             label='Valeur moyenne')
    plt.legend(loc=0)
    plt.xlabel(r'Nombre de faux $2$-simplexes')
    plt.ylabel(r'Nombre de r\'ealisations ')
    #plt.xlim(10, 30)
    ax4.xaxis.set_minor_locator(MultipleLocator(1))
    plt.show()

    plt.show()

    #print(nb_found_links_list)
    #print(nb_found_twosimp_list)
    #print(get_complete_list_with_pvals(
    #        '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Simplicial_complex_onefactorgraph/123_45_simplicialcomplex_100/fg.pkl',
    #        observations))
    #print(mean/1000)
    #expected_for_mean = mle_2x2_ind(mean/1000)
    #print('Mean [0, 1] : ', mean/1000, chisq_test_here(mean/1000, expected_for_mean))
    print('Nb VP 1simplices, Total number of links : ', np.sum(total_nb_vp_link_list), np.sum(total_nb_fp_link_list))
    print('Nb FP 2simplices, Total number of 2simplices : ', np.sum(total_nb_vp_2simp_list), np.sum(total_nb_fp_2simp_list))
    print('Pvalues 1-simplex (max VP, min VP, max FP, min FP) : ', max(pval_list_vp_link), min(pval_list_vp_link), max(pval_list_fp_link), min(pval_list_fp_link))
    print('Pvalues 2-simplex (max VP, min VP, max FP, min FP) : ', max(pval_list_vp_2simp), min(pval_list_vp_2simp), max(pval_list_fp_2simp),min(pval_list_fp_2simp))


    exit()


    print(len(np.where(np.array(link_count) == 0)[0]), len(np.where(np.array(link_count) == 1)[0]),
          len(np.where(np.array(link_count) == 2)[0]), len(np.where(np.array(link_count) == 3)[0]))

    list_of_pval_list.append(copy.deepcopy(pval_list))
    list_of_compte_list = []
    # print(list_of_pval_list)
    # exit()
    print(np.amax(np.array(list_of_pval_list)))
    plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001), ls='--', color='#00a1ffff', label=r'$y = \alpha$')
    for pval_list in list_of_pval_list:
        # print(pval_list)
        comptelist = []
        for alpha in np.arange(0, 0.1001, 0.001):
            compte = 0
            for pval in pval_list:
                if pval > alpha:
                    compte += 1
                    # print(pval)
            comptelist.append(compte)
        # print(comptelist)
        list_of_compte_list.append(np.array(comptelist) / 1000)
        # plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist)/1000)

    plt.plot(np.arange(0, 0.101, 0.001), np.mean(np.array(list_of_compte_list), axis=0), color='#ff7f00ff',
             linewidth='2', label='Proportion de rejet')
    plt.fill_between(np.arange(0, 0.101, 0.001),
                     np.mean(np.array(list_of_compte_list), axis=0) + np.std(np.array(list_of_compte_list), axis=0),
                     np.mean(np.array(list_of_compte_list), axis=0) - np.std(np.array(list_of_compte_list), axis=0),
                     color='#ff7f00ff', alpha=0.2)
    plt.legend(loc=0)
    plt.ylabel(r"Proportion d\textquotesingle erreur de type $1$")
    plt.xlabel(r'$\alpha$')
    plt.show()

    pval_list = []
    for i in np.arange(0, 10000, 1):

        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/three_dep_triangle_to_simplex_clean/data_100_2/bipartite_' + str(
                i) + '.npy')

        analyser = Analyser(bam,
                            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/three_dep_triangle_to_simplex_clean/factorgraph_three-ind')
        if analyser.analyse_asymptotic_for_triangle(0.01) == 3:
            cont_table = get_cont_cube(1, 2, 0, bam)

            exp = iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_table)

            pval = chisq_test_here(cont_table, exp, df=1)[1]

            pval_list.append(pval)

    print(pval_list)

    # exit()

    # plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001), ls='--', color='#00a1ffff', label=r'$y = \alpha$')
    comptelist = []
    for alpha in np.arange(0.001, 0.1001, 0.001):
        compte = 0
        for pval in pval_list:
            if pval < alpha:
                compte += 1
                print(pval)
        comptelist.append(compte)
    print(comptelist)
    comptelist = np.array(comptelist) / 5763
    # plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist)/1000)
    rc('text', usetex=True)
    rc('font', size=16)
    plt.plot(np.arange(0.001, 0.101, 0.001), comptelist, color='#ff7f00ff',
             linewidth='2', label='Proportion de rejet')

    plt.legend(loc=0)
    plt.ylabel(r"Proportion d\textquotesingle erreur de type $2$")
    plt.xlabel(r'$\alpha$')
    plt.show()
    exit()

    exit()
    comptelist = []
    for alpha in np.arange(0, 0.1001, 0.001):
        compte = 0
        for pval in pval_list:
            if pval < alpha:
                compte += 1
                print(pval)
        comptelist.append(compte)
    print(comptelist)
    plt.plot(np.arange(0, 0.101, 0.001), np.arange(0, 0.101, 0.001))
    plt.plot(np.arange(0, 0.101, 0.001), np.array(comptelist) / 1000)

    plt.show()
    compte = 0
    for pval in pval_list:
        if pval < 0.01:
            compte += 1
            print(pval)
    print('Compte under 0.01 : ', compte)

    distance_list.sort()
    print(distance_list)

    plt.figure(1)
    n, b, p = plt.hist(distance_list, bins=99)
    plt.xlabel('distance')
    plt.ylabel('count')
    # plt.xlim(0, 1)
    plt.show()

    # pval_list.sort()
    # print(pval_list)
    # print(min(pval_list))
    # print('Compte = ', compte)

    np.save('pval_list_1000', np.array(pval_list))
    print(pval_list)
    # exit()
    # pval_list = np.load('pval_list_100.npy')
    plt.figure(1)
    n, b, p = plt.hist(pval_list, bins=np.arange(0, 1, 0.01))
    plt.xlabel('p-value')
    plt.ylabel('count')
    plt.xlim(0, 1)
    plt.show()

    sum = np.array([[0.0, 0.0], [0.0, 0.0]])
    for i in range(1000):
        bam = np.load(
            '/home/xavier/Documents/Projet/Betti_scm/Clean_factorgraph/Two_co_clean/data_1000/bipartite_' + str(
                i) + '.npy')

        cont_table = get_cont_table(0, 1, bam)

        print(cont_table)

        sum += cont_table

    print(sum / (1000 * 100))
